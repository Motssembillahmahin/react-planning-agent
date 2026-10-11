"""Agent configuration. Each guard can be disabled independently by setting it to None."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from react_agent.tools.base import RetryPolicy


class AgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Termination guards (None disables a guard; max_iterations is always on).
    max_iterations: int = Field(default=12, ge=1)
    token_budget: int | None = Field(default=60_000, ge=1)
    timeout_s: float | None = Field(default=120.0, gt=0)
    repeat_window: int = Field(default=6, ge=1)
    repeat_threshold: int | None = Field(default=3, ge=2)
    no_progress_patience: int | None = Field(default=3, ge=1)
    tool_error_budget: int | None = Field(default=3, ge=1)

    # Self-critique.
    critic_enabled: bool = True
    max_critic_rejections: int = Field(default=2, ge=1)

    # Tool execution.
    tool_max_retries: int = Field(default=2, ge=0)
    tool_backoff_s: float = Field(default=0.05, ge=0)
    tool_timeout_s: float | None = Field(default=10.0, gt=0)

    # Prompt size control: only the last N turns are sent to the model.
    history_window: int = Field(default=20, ge=1)

    def retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_retries=self.tool_max_retries,
            backoff_s=self.tool_backoff_s,
            timeout_s=self.tool_timeout_s,
        )
