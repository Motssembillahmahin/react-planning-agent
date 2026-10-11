import pytest
from pydantic import ValidationError

from react_agent.llm import LLMError, Message, ParseError, ScriptedLLM, parse_model_step
from react_agent.llm.base import LLMResponse, estimate_tokens
from react_agent.types import Action, ModelStep

MSG = [Message(role="user", content="hello world, this is a prompt")]


def test_replays_in_order_and_serialises_dicts() -> None:
    llm = ScriptedLLM(["a", {"final_answer": "b"}])
    assert llm.complete(MSG).text == "a"
    assert llm.complete(MSG, purpose="critic").text == '{"final_answer": "b"}'
    assert llm.calls == 2
    assert llm.purposes == ["step", "critic"]


def test_exhausted_error() -> None:
    llm = ScriptedLLM(["a"])
    llm.complete(MSG)
    with pytest.raises(LLMError, match="exhausted"):
        llm.complete(MSG)


def test_empty_script_always_errors_even_when_cycling() -> None:
    with pytest.raises(LLMError):
        ScriptedLLM([], on_exhausted="cycle").complete(MSG)


def test_repeat_last_and_cycle() -> None:
    rep = ScriptedLLM(["a", "b"], on_exhausted="repeat_last")
    assert [rep.complete(MSG).text for _ in range(4)] == ["a", "b", "b", "b"]
    cyc = ScriptedLLM(["a", "b"], on_exhausted="cycle")
    assert [cyc.complete(MSG).text for _ in range(5)] == ["a", "b", "a", "b", "a"]


def test_token_estimates_are_deterministic() -> None:
    r = ScriptedLLM(["x" * 40]).complete(MSG)
    assert r.output_tokens == 10
    assert r.input_tokens == estimate_tokens(MSG[0].content)
    assert r.total_tokens == r.input_tokens + r.output_tokens
    assert estimate_tokens("") == 0
    assert estimate_tokens("ab") == 1
    assert LLMResponse(text="t").total_tokens == 0


def test_parse_action_with_prose_and_fences() -> None:
    text = 'Sure!\n```json\n{"thought": "t", "action": {"tool": "calc", "args": {"x": 1}}}\n```'
    step = parse_model_step(text)
    assert step.action == Action(tool="calc", args={"x": 1})
    assert step.thought == "t"


def test_parse_skips_broken_brace_before_valid_object() -> None:
    step = parse_model_step('{not json} then {"final_answer": "42"}')
    assert step.final_answer == "42"


@pytest.mark.parametrize(
    "text",
    [
        "no json here",
        "{broken",
        '{"thought": "only thought"}',
        '{"final_answer": "a", "abstain": "b"}',
        '{"action": {"tool": ""}}',
    ],
)
def test_parse_errors(text: str) -> None:
    with pytest.raises(ParseError):
        parse_model_step(text)


def test_model_step_requires_exactly_one() -> None:
    with pytest.raises(ValidationError):
        ModelStep()
    assert ModelStep(abstain="unknown").abstain == "unknown"
