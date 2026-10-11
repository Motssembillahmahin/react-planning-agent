"""OpenAICompatibleLLM: any Chat Completions endpoint (OpenAI, Ollama, vLLM, ...).

The key comes from OPENAI_API_KEY; the endpoint from `base_url` or OPENAI_BASE_URL.
For Ollama use base_url="http://localhost:11434/v1" (no key needed; a placeholder is sent).
The SDK is an optional extra: `uv sync --extra openai`.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from react_agent.llm.base import LLMError, LLMResponse, Message

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAICompatibleLLM:
    name = "openai-compatible"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str | None = None,
        client: Any = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        if client is None:
            base_url = base_url or os.environ.get("OPENAI_BASE_URL")
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key and not base_url:
                raise LLMError("OPENAI_API_KEY is not set (or pass base_url for a local server)")
            try:
                sdk = importlib.import_module("openai")
            except ImportError as exc:
                raise LLMError("install the extra: uv sync --extra openai") from exc
            client = sdk.OpenAI(api_key=api_key or "local", base_url=base_url, timeout=timeout_s)
        self._client = client

    def complete(self, messages: list[Message], *, purpose: str = "step") -> LLMResponse:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
            )
        except Exception as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        if not response.choices:
            raise LLMError("empty response")
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=response.choices[0].message.content or "",
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )
