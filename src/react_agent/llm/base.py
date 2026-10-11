"""The LLM interface used by the agent and the critic."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str


class LLMResponse(BaseModel):
    text: str
    input_tokens: int = Field(ge=0, default=0)
    output_tokens: int = Field(ge=0, default=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMError(RuntimeError):
    """Raised by an LLM implementation when it cannot produce a response."""


class LLM(Protocol):
    """Anything that turns a conversation into a text completion."""

    name: str

    def complete(self, messages: list[Message], *, purpose: str = "step") -> LLMResponse:
        """Return the model's reply. `purpose` is 'step' or 'critic'."""
        ...


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 characters per token)."""
    return max(1, len(text) // 4) if text else 0
