"""Property: for arbitrary model outputs and tool outcomes, the loop terminates within
max_iterations + rejection cap model calls and always returns a valid result."""

from itertools import cycle
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, ConfigDict

from react_agent.agent import ReActAgent
from react_agent.config import AgentConfig
from react_agent.critic import Critic, LLMCritic
from react_agent.llm import ScriptedLLM
from react_agent.tools import Tool, ToolError, ToolInputError, ToolRegistry, ToolTimeoutError
from react_agent.trace import Tracer
from react_agent.types import AgentResult, Phase, Status

P = Phase
ALLOWED = {
    (None, P.OBSERVE), (P.OBSERVE, P.THINK), (P.OBSERVE, P.DEGRADED),
    (P.THINK, P.ACT), (P.THINK, P.REFLECT), (P.THINK, P.CRITIQUE), (P.THINK, P.DONE),
    (P.THINK, P.DEGRADED), (P.ACT, P.REFLECT), (P.ACT, P.DEGRADED), (P.REFLECT, P.OBSERVE),
    (P.REFLECT, P.DEGRADED), (P.CRITIQUE, P.DONE), (P.CRITIQUE, P.DEGRADED),
    (P.CRITIQUE, P.OBSERVE),
}  # fmt: skip

words = st.text(alphabet="abc xyz 0123456789.", max_size=12)
args = st.dictionaries(
    st.sampled_from(["q", "x", "n"]), st.one_of(words, st.integers(-5, 5)), max_size=2
)
action = st.builds(
    lambda tool, a, t: {"thought": t, "action": {"tool": tool, "args": a}},
    st.sampled_from(["tool_a", "tool_b", "missing"]),
    args,
    words,
)
model_output = st.one_of(
    action,
    st.builds(lambda a: {"final_answer": a}, words),
    st.builds(lambda a: {"abstain": a}, words),
    st.builds(
        lambda v, r: {"verdict": v, "reason": r}, st.sampled_from(["accept", "reject"]), words
    ),
    st.text(max_size=30),
    st.just({"action": {"tool": ""}}),
    st.just({"final_answer": "1", "abstain": "2"}),
)
tool_outcome = st.one_of(
    st.builds(lambda w: ("ok", w), words),
    st.just(("error", "")),
    st.just(("input_error", "")),
    st.just(("timeout", "")),
    st.just(("crash", "")),
)


class AnyArgs(BaseModel):
    model_config = ConfigDict(extra="allow")


def fake_tool(name: str, outcomes: list[tuple[str, str]]) -> Tool:
    stream = cycle(outcomes or [("ok", "")])

    def fn(_: BaseModel) -> str:
        kind, text = next(stream)
        if kind == "error":
            raise ToolError("transient")
        if kind == "input_error":
            raise ToolInputError("bad input")
        if kind == "timeout":
            raise ToolTimeoutError("timed out")
        if kind == "crash":
            raise RuntimeError("crash")
        return text

    return Tool(name=name, description="fake", args_model=AnyArgs, fn=fn)


configs = st.builds(
    AgentConfig,
    max_iterations=st.integers(1, 15),
    max_critic_rejections=st.integers(1, 4),
    repeat_threshold=st.one_of(st.none(), st.integers(2, 4)),
    repeat_window=st.integers(1, 8),
    no_progress_patience=st.one_of(st.none(), st.integers(1, 4)),
    tool_error_budget=st.one_of(st.none(), st.integers(1, 4)),
    token_budget=st.one_of(st.none(), st.integers(1, 5000)),
    critic_enabled=st.booleans(),
    tool_max_retries=st.integers(0, 2),
    tool_backoff_s=st.just(0.0),
    tool_timeout_s=st.none(),
)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    script=st.lists(model_output, max_size=25),
    on_exhausted=st.sampled_from(["error", "repeat_last", "cycle"]),
    outcomes_a=st.lists(tool_outcome, max_size=6),
    outcomes_b=st.lists(tool_outcome, max_size=6),
    config=configs,
    llm_critic=st.booleans(),
)
def test_loop_always_terminates_with_valid_result(
    script: list[Any],
    on_exhausted: Any,
    outcomes_a: list[tuple[str, str]],
    outcomes_b: list[tuple[str, str]],
    config: AgentConfig,
    llm_critic: bool,
) -> None:
    llm = ScriptedLLM(script, on_exhausted=on_exhausted)
    registry = ToolRegistry([fake_tool("tool_a", outcomes_a), fake_tool("tool_b", outcomes_b)])
    registry.set_fallback("tool_a", "tool_b")
    critic: Critic | None = LLMCritic(llm) if llm_critic else None
    agent = ReActAgent(llm, registry, config, critic=critic, sleep=lambda _: None)

    captured: list[Tracer] = []
    original = Tracer.__init__

    def spy(self: Tracer, *a: Any, **kw: Any) -> None:
        original(self, *a, **kw)
        captured.append(self)

    Tracer.__init__ = spy  # type: ignore[method-assign]
    try:
        result = agent.run("property task 42")
    finally:
        Tracer.__init__ = original  # type: ignore[method-assign]

    # Always a valid result object, never an exception.
    assert isinstance(result, AgentResult)
    AgentResult.model_validate(result.model_dump())
    # Bounded: model calls for steps + critic calls never exceed the documented bound.
    assert result.iterations <= config.max_iterations
    assert llm.calls <= config.max_iterations + config.max_critic_rejections
    assert result.critic_rejections <= config.max_critic_rejections
    # Status/answer consistency.
    if result.status is Status.SUCCESS:
        assert result.answer is not None
        assert result.guard is None
    else:
        assert result.guard is not None
    if result.status is Status.ABSTAINED:
        assert result.answer is None
    # Every transition is a legal edge of the state machine and the run ends in DONE/DEGRADED.
    (tracer,) = captured
    edges = [(Phase(a) if a else None, Phase(b)) for a, b in tracer.transitions()]
    assert set(edges) <= ALLOWED
    assert edges[-1][1] in (Phase.DONE, Phase.DEGRADED)
    assert tracer.events[-1]["event"] == "result"
