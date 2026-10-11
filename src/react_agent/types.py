"""Core data models shared by the loop, tools, critic, and evaluation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Phase(StrEnum):
    """Phases of the agent state machine."""

    OBSERVE = "OBSERVE"
    THINK = "THINK"
    ACT = "ACT"
    REFLECT = "REFLECT"
    CRITIQUE = "CRITIQUE"
    DONE = "DONE"
    DEGRADED = "DEGRADED"


class Status(StrEnum):
    """Outcome of a run.

    - success: an answer was accepted (by the critic, if enabled).
    - partial: a guard stopped the run, but a candidate answer exists.
    - abstained: the agent declined to answer (model abstained, critic cap,
      or it stopped making progress without any candidate answer).
    - failed: resources or infrastructure ran out before any answer.
    """

    SUCCESS = "success"
    PARTIAL = "partial"
    ABSTAINED = "abstained"
    FAILED = "failed"


class GuardName(StrEnum):
    """Every reason the loop can stop other than an accepted answer."""

    MAX_ITERATIONS = "max_iterations"
    TOKEN_BUDGET = "token_budget"
    TIMEOUT = "timeout"
    REPEATED_ACTION = "repeated_action"
    NO_PROGRESS = "no_progress"
    TOOL_ERROR_BUDGET = "tool_error_budget"
    CRITIC_REJECTIONS = "critic_rejections"
    MODEL_ABSTAINED = "model_abstained"
    LLM_ERROR = "llm_error"
    INTERNAL_ERROR = "internal_error"
    SAFETY_CAP = "safety_cap"


class Action(BaseModel):
    """A tool invocation proposed by the model."""

    model_config = ConfigDict(frozen=True)

    tool: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)


class ModelStep(BaseModel):
    """One parsed model output: exactly one of action, final_answer, abstain."""

    thought: str = ""
    action: Action | None = None
    final_answer: str | None = None
    abstain: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> ModelStep:
        chosen = [x for x in (self.action, self.final_answer, self.abstain) if x is not None]
        if len(chosen) != 1:
            raise ValueError("exactly one of 'action', 'final_answer', 'abstain' is required")
        return self


class ToolResult(BaseModel):
    """Outcome of executing an action, after retries and fallback."""

    tool: str
    ok: bool
    output: str = ""
    error: str | None = None
    attempts: int = 1
    used_fallback: str | None = None

    def as_observation(self) -> str:
        if self.ok:
            prefix = f"[{self.used_fallback} (fallback)] " if self.used_fallback else ""
            return f"{prefix}{self.output}"
        return f"ERROR from {self.tool}: {self.error}"


class Critique(BaseModel):
    """Verdict of the critic on a proposed final answer."""

    accepted: bool
    reason: str


class AgentResult(BaseModel):
    """Structured result returned by every run; the loop never raises."""

    status: Status
    answer: str | None = None
    guard: GuardName | None = None
    reason: str
    iterations: int = Field(ge=0)
    tokens_used: int = Field(ge=0, default=0)
    llm_calls: int = Field(ge=0, default=0)
    tool_errors: int = Field(ge=0, default=0)
    critic_rejections: int = Field(ge=0, default=0)
    elapsed_s: float = Field(ge=0.0, default=0.0)
    trace_path: str | None = None
