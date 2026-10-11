"""Provider classes are tested with fake clients only; no network, no paid API."""

import sys
import types
from types import SimpleNamespace as Ns
from typing import Any

import pytest

from react_agent.llm import LLMError, Message
from react_agent.llm.anthropic_llm import FALLBACK_BETA, AnthropicLLM, merge_turns
from react_agent.llm.openai_llm import OpenAICompatibleLLM
from react_agent.llm.providers import available_provider, make_llm

MSGS = [
    Message(role="system", content="sys"),
    Message(role="user", content="task"),
    Message(role="user", content="(2 omitted)"),
    Message(role="assistant", content="a"),
]


class FakeAnthropicMessages:
    def __init__(self, response: Any = None, exc: Exception | None = None) -> None:
        self.response = response
        self.exc = exc
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        return self.response


def anthropic_response(stop: str = "end_turn") -> Any:
    return Ns(
        stop_reason=stop,
        content=[Ns(type="thinking", thinking=""), Ns(type="text", text='{"final_answer": "4"}')],
        usage=Ns(input_tokens=11, output_tokens=5),
    )


def test_merge_turns() -> None:
    system, turns = merge_turns(MSGS)
    assert system == "sys"
    assert turns == [
        {"role": "user", "content": "task\n\n(2 omitted)"},
        {"role": "assistant", "content": "a"},
    ]


def test_anthropic_with_fallbacks() -> None:
    beta = FakeAnthropicMessages(anthropic_response())
    client = Ns(beta=Ns(messages=beta), messages=FakeAnthropicMessages())
    r = AnthropicLLM(client=client).complete(MSGS)
    assert r.text == '{"final_answer": "4"}'
    assert (r.input_tokens, r.output_tokens) == (11, 5)
    assert beta.kwargs["betas"] == [FALLBACK_BETA]
    assert beta.kwargs["fallbacks"] == "default"
    assert beta.kwargs["model"] == "claude-opus-5"
    assert beta.kwargs["system"] == "sys"


def test_anthropic_plain_refusal_and_errors() -> None:
    plain = FakeAnthropicMessages(anthropic_response())
    llm = AnthropicLLM("m", client=Ns(messages=plain), refusal_fallbacks=False)
    llm.complete([Message(role="user", content="hi")])
    assert "system" not in plain.kwargs
    refused = AnthropicLLM(
        client=Ns(messages=FakeAnthropicMessages(anthropic_response("refusal"))),
        refusal_fallbacks=False,
    )
    with pytest.raises(LLMError, match="refused"):
        refused.complete(MSGS)
    broken = AnthropicLLM(
        client=Ns(messages=FakeAnthropicMessages(exc=RuntimeError("down"))),
        refusal_fallbacks=False,
    )
    with pytest.raises(LLMError, match="RuntimeError: down"):
        broken.complete(MSGS)


def test_anthropic_env_and_sdk_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        AnthropicLLM()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-real-key")
    monkeypatch.setitem(sys.modules, "anthropic", None)  # simulate missing extra
    with pytest.raises(LLMError, match="uv sync --extra anthropic"):
        AnthropicLLM()
    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda timeout: Ns(timeout=timeout)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    assert AnthropicLLM(timeout_s=5)._client.timeout == 5


class FakeCompletions:
    def __init__(self, response: Any = None, exc: Exception | None = None) -> None:
        self.response = response
        self.exc = exc
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        return self.response


def openai_client(response: Any = None, exc: Exception | None = None) -> Any:
    return Ns(chat=Ns(completions=FakeCompletions(response, exc)))


def test_openai_compatible() -> None:
    resp = Ns(
        choices=[Ns(message=Ns(content="hello"))],
        usage=Ns(prompt_tokens=7, completion_tokens=2),
    )
    client = openai_client(resp)
    r = OpenAICompatibleLLM("llama3.1", client=client).complete(MSGS)
    assert (r.text, r.input_tokens, r.output_tokens) == ("hello", 7, 2)
    assert client.chat.completions.kwargs["messages"][0] == {"role": "system", "content": "sys"}
    no_usage = OpenAICompatibleLLM(client=openai_client(Ns(choices=[Ns(message=Ns(content=None))])))
    assert no_usage.complete(MSGS).text == ""
    with pytest.raises(LLMError, match="empty"):
        OpenAICompatibleLLM(client=openai_client(Ns(choices=[]))).complete(MSGS)
    with pytest.raises(LLMError, match="ValueError"):
        OpenAICompatibleLLM(client=openai_client(exc=ValueError("x"))).complete(MSGS)


def test_openai_env_and_sdk_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        OpenAICompatibleLLM()
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(LLMError, match="uv sync --extra openai"):
        OpenAICompatibleLLM(base_url="http://localhost:11434/v1")
    fake = types.ModuleType("openai")
    fake.OpenAI = lambda **kw: Ns(**kw)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake)
    llm = OpenAICompatibleLLM(base_url="http://localhost:11434/v1")
    assert llm._client.api_key == "local"
    assert llm._client.base_url == "http://localhost:11434/v1"


def test_make_llm_and_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    assert available_provider() is None
    with pytest.raises(LLMError, match="no provider key"):
        make_llm("auto")
    with pytest.raises(LLMError, match="unknown provider"):
        make_llm("bogus")
    fake_a = types.ModuleType("anthropic")
    fake_a.Anthropic = lambda timeout: Ns()  # type: ignore[attr-defined]
    fake_o = types.ModuleType("openai")
    fake_o.OpenAI = lambda **kw: Ns(**kw)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_a)
    monkeypatch.setitem(sys.modules, "openai", fake_o)
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    assert available_provider() == "openai"
    assert make_llm("auto").name == "openai-compatible"
    ollama = make_llm("ollama")
    assert isinstance(ollama, OpenAICompatibleLLM)
    assert ollama.model == "llama3.1"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-real-key")
    assert available_provider() == "anthropic"
    assert make_llm("auto").name == "anthropic"
    assert make_llm("anthropic", "claude-sonnet-5").model == "claude-sonnet-5"  # type: ignore[attr-defined]
