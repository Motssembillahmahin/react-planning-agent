"""AnthropicLLM: the step/critic model backed by the Anthropic Messages API.

The API key is read from the ANTHROPIC_API_KEY environment variable (never from code or files).
The SDK is an optional extra: `uv sync --extra anthropic`.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from react_agent.llm.base import LLMError, LLMResponse, Message

DEFAULT_MODEL = "claude-opus-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def merge_turns(messages: list[Message]) -> tuple[str, list[dict[str, str]]]:
    """Split out system text and merge consecutive same-role turns."""
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    turns: list[dict[str, str]] = []
    for m in messages:
        if m.role == "system":
            continue
        if turns and turns[-1]["role"] == m.role:
            turns[-1]["content"] += "\n\n" + m.content
        else:
            turns.append({"role": m.role, "content": m.content})
    return system, turns


class AnthropicLLM:
    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        client: Any = None,
        max_tokens: int = 16000,
        timeout_s: float = 120.0,
        refusal_fallbacks: bool = True,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.refusal_fallbacks = refusal_fallbacks
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise LLMError("ANTHROPIC_API_KEY is not set")
            try:
                sdk = importlib.import_module("anthropic")
            except ImportError as exc:
                raise LLMError("install the extra: uv sync --extra anthropic") from exc
            client = sdk.Anthropic(timeout=timeout_s)  # reads ANTHROPIC_API_KEY itself
        self._client = client

    def complete(self, messages: list[Message], *, purpose: str = "step") -> LLMResponse:
        system, turns = merge_turns(messages)
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": turns,
        }
        if system:
            params["system"] = system
        try:
            if self.refusal_fallbacks:
                response = self._client.beta.messages.create(
                    betas=[FALLBACK_BETA], fallbacks="default", **params
                )
            else:
                response = self._client.messages.create(**params)
        except Exception as exc:  # SDK errors become LLMError; the agent degrades on it
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        if getattr(response, "stop_reason", None) == "refusal":
            raise LLMError("model refused the request")
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        usage = response.usage
        return LLMResponse(
            text=text,
            input_tokens=int(usage.input_tokens or 0),
            output_tokens=int(usage.output_tokens or 0),
        )
