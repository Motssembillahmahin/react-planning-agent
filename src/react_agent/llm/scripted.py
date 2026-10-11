"""ScriptedLLM: a deterministic model that replays a fixed list of responses."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal

from react_agent.llm.base import LLMError, LLMResponse, Message, estimate_tokens

OnExhausted = Literal["error", "repeat_last", "cycle"]


class ScriptedLLM:
    """Replays `responses` in order, one per `complete()` call, regardless of purpose.

    Dict responses are serialised to JSON. When the script runs out, behaviour is set
    by `on_exhausted`: raise LLMError ("error"), repeat the last response
    ("repeat_last"), or start over ("cycle"). Token counts are deterministic
    estimates so budgets can be tested.
    """

    name = "scripted"

    def __init__(
        self,
        responses: Sequence[str | dict[str, Any]],
        *,
        on_exhausted: OnExhausted = "error",
    ) -> None:
        self._responses = [r if isinstance(r, str) else json.dumps(r) for r in responses]
        self._on_exhausted: OnExhausted = on_exhausted
        self.calls = 0
        self.purposes: list[str] = []

    def complete(self, messages: list[Message], *, purpose: str = "step") -> LLMResponse:
        index = self.calls
        self.calls += 1
        self.purposes.append(purpose)
        n = len(self._responses)
        if index >= n:
            if n == 0 or self._on_exhausted == "error":
                raise LLMError(f"script exhausted after {n} responses")
            index = n - 1 if self._on_exhausted == "repeat_last" else index % n
        text = self._responses[index]
        prompt_tokens = sum(estimate_tokens(m.content) for m in messages)
        return LLMResponse(
            text=text, input_tokens=prompt_tokens, output_tokens=estimate_tokens(text)
        )
