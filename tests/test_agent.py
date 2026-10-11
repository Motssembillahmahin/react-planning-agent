from pathlib import Path
from typing import Any

import pytest

from react_agent.agent import ReActAgent, status_for
from react_agent.config import AgentConfig
from react_agent.critic import LLMCritic
from react_agent.llm import LLMError, LLMResponse, Message, ScriptedLLM
from react_agent.tools import (
    ToolRegistry,
    always_fail,
    calculator_tool,
    default_registry,
    search_tool,
)
from react_agent.trace import read_trace
from react_agent.types import GuardName, Phase, Status

CALC = {"thought": "compute", "action": {"tool": "calculator", "args": {"expression": "6*7"}}}
FAST = AgentConfig(tool_backoff_s=0.0, tool_timeout_s=None)


def agent(script: list[Any], config: AgentConfig = FAST, **kw: Any) -> ReActAgent:
    on_exhausted = kw.pop("on_exhausted", "error")
    tools = kw.pop("tools", None) or default_registry()
    return ReActAgent(ScriptedLLM(script, on_exhausted=on_exhausted), tools, config, **kw)


def test_success_path_and_trace(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    res = agent([CALC, {"final_answer": "42"}], trace_path=path).run("What is 6*7?")
    assert res.status is Status.SUCCESS
    assert res.answer == "42"
    assert res.guard is None
    assert res.iterations == 2
    assert res.tokens_used > 0
    assert res.trace_path == str(path)
    events = read_trace(path)
    transitions = [(e["from"], e["to"]) for e in events if e["event"] == "transition"]
    assert transitions == [
        (None, "OBSERVE"), ("OBSERVE", "THINK"), ("THINK", "ACT"), ("ACT", "REFLECT"),
        ("REFLECT", "OBSERVE"), ("OBSERVE", "THINK"), ("THINK", "CRITIQUE"), ("CRITIQUE", "DONE"),
    ]  # fmt: skip
    assert events[0]["event"] == "start"
    assert events[-1]["event"] == "result"
    assert events[-1]["status"] == "success"


def test_critic_rejection_feeds_back_then_accepts() -> None:
    a = agent([{"final_answer": "40"}, CALC, {"final_answer": "42"}])
    res = a.run("What is 6*7?")
    assert res.status is Status.SUCCESS
    assert res.answer == "42"
    assert res.critic_rejections == 1


def test_critic_feedback_is_shown_to_model() -> None:
    seen: list[str] = []

    class Spy(ScriptedLLM):
        def complete(self, messages: list[Message], *, purpose: str = "step") -> LLMResponse:
            seen.append(messages[-1].content)
            return super().complete(messages, purpose=purpose)

    ReActAgent(Spy([{"final_answer": "40"}, {"abstain": "x"}]), default_registry(), FAST).run("t")
    assert "Critic rejected the answer" in seen[1]


def test_critic_rejection_cap_abstains() -> None:
    res = agent([{"final_answer": "1"}, {"final_answer": "2"}, {"final_answer": "3"}]).run("q")
    assert res.status is Status.ABSTAINED
    assert res.guard is GuardName.CRITIC_REJECTIONS
    assert res.critic_rejections == 2
    assert res.answer is None
    assert "'2'" in res.reason


def test_rejection_cap_is_configurable() -> None:
    cfg = FAST.model_copy(update={"max_critic_rejections": 1})
    res = agent([{"final_answer": "1"}], cfg).run("q")
    assert res.guard is GuardName.CRITIC_REJECTIONS
    assert res.critic_rejections == 1


def test_critic_disabled_accepts_anything() -> None:
    res = agent(
        [{"final_answer": "made up"}], FAST.model_copy(update={"critic_enabled": False})
    ).run("q")
    assert res.status is Status.SUCCESS
    assert res.reason == "critic disabled"


def test_llm_critic_consumes_tokens_and_budget() -> None:
    llm = ScriptedLLM([CALC, {"final_answer": "42"}, {"verdict": "accept", "reason": "ok"}])
    a = ReActAgent(llm, default_registry(), FAST, critic=LLMCritic(llm))
    res = a.run("6*7?")
    assert res.status is Status.SUCCESS
    assert llm.purposes == ["step", "step", "critic"]


def test_model_abstains() -> None:
    res = agent([{"abstain": "unknowable"}]).run("q")
    assert res.status is Status.ABSTAINED
    assert res.guard is GuardName.MODEL_ABSTAINED
    assert res.reason == "unknowable"


def test_max_iterations_guard() -> None:
    script = [
        {"action": {"tool": "calculator", "args": {"expression": f"{i}+1"}}} for i in range(10)
    ]
    cfg = FAST.model_copy(update={"max_iterations": 4})
    res = agent(script, cfg).run("q")
    assert res.guard is GuardName.MAX_ITERATIONS
    assert res.status is Status.FAILED
    assert res.iterations == 4


def test_repeated_action_guard_trips_before_third_execution() -> None:
    res = agent([CALC], on_exhausted="repeat_last").run("q")
    assert res.guard is GuardName.REPEATED_ACTION
    assert res.status is Status.ABSTAINED
    assert res.iterations == 3


def test_no_progress_guard_on_parse_errors() -> None:
    res = agent(["not json"], on_exhausted="repeat_last").run("q")
    assert res.guard is GuardName.NO_PROGRESS
    assert res.iterations == 3


def test_no_progress_when_repeat_detector_disabled() -> None:
    cfg = FAST.model_copy(update={"repeat_threshold": None})
    res = agent([CALC], cfg, on_exhausted="repeat_last").run("q")
    assert res.guard is GuardName.NO_PROGRESS
    assert res.iterations == 4  # first call is new information, then 3 without


def test_tool_error_budget_guard() -> None:
    reg = ToolRegistry([always_fail(calculator_tool())])
    script = [{"action": {"tool": "calculator", "args": {"expression": f"{i}"}}} for i in range(5)]
    res = agent(script, tools=reg).run("q")
    assert res.guard is GuardName.TOOL_ERROR_BUDGET
    assert res.status is Status.FAILED
    assert res.tool_errors == 3


def test_token_budget_guard() -> None:
    cfg = FAST.model_copy(update={"token_budget": 50})
    res = agent([CALC, CALC], cfg).run("q")
    assert res.guard is GuardName.TOKEN_BUDGET
    assert res.iterations == 1


def test_timeout_guard_with_fake_clock() -> None:
    now = [0.0]

    def clock() -> float:
        now[0] += 10.0
        return now[0]

    cfg = FAST.model_copy(update={"timeout_s": 25.0})
    res = agent([CALC] * 5, cfg, clock=clock).run("q")
    assert res.guard is GuardName.TIMEOUT
    assert res.status is Status.FAILED


def test_partial_when_guard_trips_after_rejected_answer() -> None:
    script = [{"final_answer": "41"}, CALC, CALC, CALC]
    res = agent(script).run("q")
    assert res.status is Status.PARTIAL
    assert res.answer == "41"
    assert res.guard is GuardName.REPEATED_ACTION


def test_llm_error_degrades() -> None:
    res = agent([CALC]).run("q")
    assert res.guard is GuardName.LLM_ERROR
    assert res.status is Status.FAILED
    assert "exhausted" in res.reason


def test_tool_fallback_inside_agent() -> None:
    reg = ToolRegistry()
    reg.register(always_fail(calculator_tool()), fallback="backup")
    reg.register(calculator_tool("backup"))
    res = agent([CALC, {"final_answer": "42"}], tools=reg).run("6*7")
    assert res.status is Status.SUCCESS
    assert res.tool_errors == 0


def test_internal_error_never_raises(tmp_path: Path) -> None:
    class Boom:
        name = "boom"

        def complete(self, messages: list[Message], *, purpose: str = "step") -> LLMResponse:
            raise ValueError("unexpected")

    res = ReActAgent(Boom(), default_registry(), FAST, trace_path=tmp_path / "t.jsonl").run("q")
    assert res.status is Status.FAILED
    assert res.guard is GuardName.INTERNAL_ERROR
    assert "ValueError: unexpected" in res.reason
    assert read_trace(tmp_path / "t.jsonl")[-2]["to"] == Phase.DEGRADED.value


def test_internal_error_with_broken_trace(tmp_path: Path) -> None:
    class Boom:
        name = "boom"

        def complete(self, messages: list[Message], *, purpose: str = "step") -> LLMResponse:
            path.unlink()
            path.mkdir()  # every later trace write now fails
            raise LLMError("x") from None

    path = tmp_path / "t.jsonl"
    res = ReActAgent(Boom(), default_registry(), FAST, trace_path=path).run("q")
    assert res.status is Status.FAILED
    assert res.guard is GuardName.INTERNAL_ERROR


@pytest.mark.parametrize(
    ("guard", "best", "status"),
    [
        (GuardName.MODEL_ABSTAINED, "a", Status.ABSTAINED),
        (GuardName.CRITIC_REJECTIONS, "a", Status.ABSTAINED),
        (GuardName.REPEATED_ACTION, "a", Status.PARTIAL),
        (GuardName.REPEATED_ACTION, None, Status.ABSTAINED),
        (GuardName.NO_PROGRESS, None, Status.ABSTAINED),
        (GuardName.MAX_ITERATIONS, None, Status.FAILED),
        (GuardName.TOOL_ERROR_BUDGET, "a", Status.PARTIAL),
        (GuardName.TIMEOUT, None, Status.FAILED),
    ],
)
def test_status_mapping(guard: GuardName, best: str | None, status: Status) -> None:
    assert status_for(guard, best) is status


def test_search_evidence_grounds_answer() -> None:
    reg = ToolRegistry([search_tool()])
    script = [
        {"action": {"tool": "search", "args": {"query": "Eiffel Tower"}}},
        {"final_answer": "330 metres"},
    ]
    res = agent(script, tools=reg).run("How tall is the Eiffel Tower?")
    assert res.status is Status.SUCCESS
