from pathlib import Path

from react_agent.baseline import NaiveReActAgent
from react_agent.llm import ScriptedLLM
from react_agent.prompts import Turn, build_critic_messages, build_messages
from react_agent.tools import ToolRegistry, always_fail, calculator_tool, default_registry
from react_agent.trace import Tracer, read_trace
from react_agent.types import GuardName, Phase, Status

CALC = {"thought": "compute", "action": {"tool": "calculator", "args": {"expression": "6*7"}}}


def test_tracer_writes_jsonl_and_tracks_transitions(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "t.jsonl"
    tracer = Tracer(path, run_id="r1", clock=lambda: 1.0)
    tracer.transition(Phase.OBSERVE)
    tracer.transition(Phase.THINK, note="x")
    tracer.emit("custom", value=3)
    events = read_trace(path)
    assert [e["seq"] for e in events] == [0, 1, 2]
    assert events[1] == {
        "seq": 1, "ts": 1.0, "run_id": "r1", "event": "transition",
        "from": "OBSERVE", "to": "THINK", "note": "x",
    }  # fmt: skip
    assert tracer.transitions() == [(None, "OBSERVE"), ("OBSERVE", "THINK")]
    assert Tracer().path is None


def test_build_messages_window_and_critic_prompt() -> None:
    turns = [Turn(f"m{i}", f"o{i}") for i in range(5)]
    msgs = build_messages("task", "- tool", turns, history_window=2)
    assert msgs[2].content == "(3 earlier steps omitted)"
    assert [m.content for m in msgs[3:]] == ["m3", "Observation: o3", "m4", "Observation: o4"]
    critic = build_critic_messages("t", "a", [])
    assert "- (none)" in critic[0].content


def test_naive_success(tmp_path: Path) -> None:
    llm = ScriptedLLM([CALC, {"final_answer": "42"}])
    agent = NaiveReActAgent(llm, default_registry(), trace_path=tmp_path / "n.jsonl")
    res = agent.run("What is 6*7?")
    assert res.status is Status.SUCCESS
    assert res.answer == "42"
    assert res.iterations == 2
    assert res.trace_path == str(tmp_path / "n.jsonl")
    assert read_trace(tmp_path / "n.jsonl")[-1]["event"] == "result"


def test_naive_hits_safety_cap_on_loop() -> None:
    llm = ScriptedLLM([CALC], on_exhausted="repeat_last")
    res = NaiveReActAgent(llm, default_registry()).run("loop")
    assert res.status is Status.FAILED
    assert res.guard is GuardName.SAFETY_CAP
    assert res.iterations == 50


def test_naive_parse_error_abstain_llm_error_and_no_fallback() -> None:
    reg = ToolRegistry()
    reg.register(always_fail(calculator_tool()), fallback="backup")
    reg.register(calculator_tool("backup"))
    llm = ScriptedLLM(["garbage", CALC, {"abstain": "cannot"}])
    res = NaiveReActAgent(llm, reg).run("t")
    assert res.status is Status.ABSTAINED
    assert res.guard is GuardName.MODEL_ABSTAINED
    assert res.tool_errors == 1  # no fallback for the baseline
    res2 = NaiveReActAgent(ScriptedLLM([]), reg).run("t")
    assert res2.status is Status.FAILED
    assert res2.guard is GuardName.LLM_ERROR
