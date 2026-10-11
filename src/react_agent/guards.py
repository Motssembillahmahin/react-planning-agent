"""Termination guards. Each is independent, configurable, and checked at fixed points of the loop.

Checkpoints:
- BEFORE_THINK: before every model call (max iterations, token budget, timeout)
- BEFORE_ACT: before executing a proposed action (repeated-action detector)
- AFTER_ACT: after a tool returns (tool error budget, timeout)
- AFTER_REFLECT: after the reflection step (no-progress detector)
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from react_agent.config import AgentConfig
from react_agent.types import Action, GuardName


class Checkpoint(StrEnum):
    BEFORE_THINK = "before_think"
    BEFORE_ACT = "before_act"
    AFTER_ACT = "after_act"
    AFTER_REFLECT = "after_reflect"


@dataclass
class LoopState:
    """Counters the guards read. The agent loop is the only writer."""

    started_at: float
    iterations: int = 0
    tokens_used: int = 0
    tool_errors: int = 0
    no_progress_streak: int = 0
    critic_rejections: int = 0
    action_history: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GuardTrip:
    guard: GuardName
    reason: str


class Guard(Protocol):
    @property
    def name(self) -> GuardName: ...

    @property
    def checkpoints(self) -> frozenset[Checkpoint]: ...

    def check(self, state: LoopState, proposed: Action | None) -> GuardTrip | None: ...


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.lower().split())
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {str(k).strip().lower(): _normalize(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize(v) for v in value]
    return value


def action_fingerprint(action: Action) -> str:
    """Stable hash of tool name + normalised args (case, whitespace, 5.0 == 5)."""
    payload = json.dumps(
        {"tool": action.tool.strip().lower(), "args": _normalize(action.args)},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class MaxIterationsGuard:
    max_iterations: int
    name: GuardName = GuardName.MAX_ITERATIONS
    checkpoints: frozenset[Checkpoint] = frozenset({Checkpoint.BEFORE_THINK})

    def check(self, state: LoopState, proposed: Action | None) -> GuardTrip | None:
        if state.iterations >= self.max_iterations:
            return GuardTrip(self.name, f"reached max_iterations={self.max_iterations}")
        return None


@dataclass(frozen=True)
class TokenBudgetGuard:
    budget: int
    name: GuardName = GuardName.TOKEN_BUDGET
    checkpoints: frozenset[Checkpoint] = frozenset({Checkpoint.BEFORE_THINK})

    def check(self, state: LoopState, proposed: Action | None) -> GuardTrip | None:
        if state.tokens_used >= self.budget:
            return GuardTrip(self.name, f"used {state.tokens_used} of {self.budget} tokens")
        return None


@dataclass(frozen=True)
class TimeoutGuard:
    timeout_s: float
    clock: Callable[[], float] = time.monotonic
    name: GuardName = GuardName.TIMEOUT
    checkpoints: frozenset[Checkpoint] = frozenset({Checkpoint.BEFORE_THINK, Checkpoint.AFTER_ACT})

    def check(self, state: LoopState, proposed: Action | None) -> GuardTrip | None:
        elapsed = self.clock() - state.started_at
        if elapsed >= self.timeout_s:
            return GuardTrip(self.name, f"{elapsed:.2f}s elapsed, limit {self.timeout_s:g}s")
        return None


@dataclass(frozen=True)
class RepeatedActionGuard:
    """Trips when the proposed action already appears threshold-1 times in the sliding window."""

    window: int
    threshold: int
    name: GuardName = GuardName.REPEATED_ACTION
    checkpoints: frozenset[Checkpoint] = frozenset({Checkpoint.BEFORE_ACT})

    def check(self, state: LoopState, proposed: Action | None) -> GuardTrip | None:
        if proposed is None:
            return None
        fp = action_fingerprint(proposed)
        recent = state.action_history[-(self.window - 1) :] if self.window > 1 else []
        count = recent.count(fp) + 1
        if count >= self.threshold:
            return GuardTrip(
                self.name,
                f"action {proposed.tool} repeated {count}x within the last {self.window} actions",
            )
        return None


@dataclass(frozen=True)
class NoProgressGuard:
    patience: int
    name: GuardName = GuardName.NO_PROGRESS
    checkpoints: frozenset[Checkpoint] = frozenset({Checkpoint.AFTER_REFLECT})

    def check(self, state: LoopState, proposed: Action | None) -> GuardTrip | None:
        if state.no_progress_streak >= self.patience:
            return GuardTrip(
                self.name, f"no new information for {state.no_progress_streak} consecutive steps"
            )
        return None


@dataclass(frozen=True)
class ToolErrorBudgetGuard:
    budget: int
    name: GuardName = GuardName.TOOL_ERROR_BUDGET
    checkpoints: frozenset[Checkpoint] = frozenset({Checkpoint.AFTER_ACT})

    def check(self, state: LoopState, proposed: Action | None) -> GuardTrip | None:
        if state.tool_errors >= self.budget:
            return GuardTrip(self.name, f"{state.tool_errors} tool failures (budget {self.budget})")
        return None


class GuardSet:
    def __init__(self, guards: list[Guard]) -> None:
        self.guards = guards

    def check(
        self, checkpoint: Checkpoint, state: LoopState, proposed: Action | None = None
    ) -> GuardTrip | None:
        for guard in self.guards:
            if checkpoint in guard.checkpoints:
                trip = guard.check(state, proposed)
                if trip is not None:
                    return trip
        return None

    def names(self) -> list[str]:
        return [g.name.value for g in self.guards]


def build_guards(config: AgentConfig, clock: Callable[[], float] = time.monotonic) -> GuardSet:
    guards: list[Guard] = [MaxIterationsGuard(config.max_iterations)]
    if config.token_budget is not None:
        guards.append(TokenBudgetGuard(config.token_budget))
    if config.timeout_s is not None:
        guards.append(TimeoutGuard(config.timeout_s, clock))
    if config.repeat_threshold is not None:
        guards.append(RepeatedActionGuard(config.repeat_window, config.repeat_threshold))
    if config.no_progress_patience is not None:
        guards.append(NoProgressGuard(config.no_progress_patience))
    if config.tool_error_budget is not None:
        guards.append(ToolErrorBudgetGuard(config.tool_error_budget))
    return GuardSet(guards)
