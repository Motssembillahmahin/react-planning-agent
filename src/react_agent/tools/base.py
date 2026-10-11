"""Tool abstraction, registry, and the executor with retry, backoff, timeout, and fallback."""

from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from react_agent.types import Action, ToolResult


class ToolError(Exception):
    """A transient tool failure; the executor may retry it."""


class ToolInputError(ToolError):
    """A deterministic failure caused by the input; never retried."""


class ToolTimeoutError(ToolError):
    """The tool did not finish within its timeout."""


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class Tool:
    """A named, typed function the agent can call.

    `fn` receives a validated instance of `args_model` and returns text.
    """

    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[Any], str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def signature(self) -> str:
        fields = ", ".join(
            f"{name}: {getattr(info.annotation, '__name__', str(info.annotation))}"
            for name, info in self.args_model.model_fields.items()
        )
        return f"{self.name}({fields}) — {self.description}"


class ToolRegistry:
    """Holds tools and optional fallbacks (primary name -> fallback name)."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._fallbacks: dict[str, str] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool, *, fallback: str | None = None) -> None:
        self._tools[tool.name] = tool
        if fallback is not None:
            self._fallbacks[tool.name] = fallback

    def set_fallback(self, primary: str, fallback: str) -> None:
        self._fallbacks[primary] = fallback

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def fallback_for(self, name: str) -> str | None:
        return self._fallbacks.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> str:
        return "\n".join(f"- {self._tools[n].signature()}" for n in self.names())


@dataclass(frozen=True)
class RetryPolicy:
    """Retry with exponential backoff, plus a per-attempt timeout (None = no timeout)."""

    max_retries: int = 2
    backoff_s: float = 0.01
    backoff_factor: float = 2.0
    timeout_s: float | None = 2.0

    def delay(self, attempt: int) -> float:
        return self.backoff_s * (self.backoff_factor**attempt)


def _call_with_timeout(fn: Callable[[], str], timeout_s: float | None) -> str:
    if timeout_s is None:
        return fn()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(fn)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError as exc:
            raise ToolTimeoutError(f"timed out after {timeout_s:g}s") from exc
    finally:
        # Python threads cannot be killed; we stop waiting and let it finish in the background.
        pool.shutdown(wait=False, cancel_futures=True)


class ToolExecutor:
    """Executes actions: validate args → attempt with retries/backoff → fallback → failure."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: RetryPolicy | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.registry = registry
        self.policy = policy or RetryPolicy()
        self._sleep = sleep

    def execute(self, action: Action) -> ToolResult:
        tool = self.registry.get(action.tool)
        if tool is None:
            known = ", ".join(self.registry.names()) or "none"
            return ToolResult(
                tool=action.tool,
                ok=False,
                error=f"unknown tool '{action.tool}' (available: {known})",
                attempts=0,
            )
        try:
            args = tool.args_model.model_validate(action.args)
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = ".".join(str(p) for p in first["loc"]) or "args"
            return ToolResult(
                tool=tool.name, ok=False, error=f"invalid args: {loc}: {first['msg']}", attempts=0
            )

        result = self._attempt(tool, args)
        if result.ok:
            return result
        fallback_name = self.registry.fallback_for(tool.name)
        fallback = self.registry.get(fallback_name) if fallback_name else None
        if fallback is None:
            return result
        try:
            fb_args = fallback.args_model.model_validate(action.args)
        except ValidationError:
            return result.model_copy(update={"error": f"{result.error}; fallback rejected args"})
        fb_result = self._attempt(fallback, fb_args)
        attempts = result.attempts + fb_result.attempts
        if fb_result.ok:
            return fb_result.model_copy(
                update={"tool": tool.name, "used_fallback": fallback.name, "attempts": attempts}
            )
        return ToolResult(
            tool=tool.name,
            ok=False,
            error=f"{result.error}; fallback {fallback.name} failed: {fb_result.error}",
            attempts=attempts,
        )

    def _attempt(self, tool: Tool, args: BaseModel) -> ToolResult:
        last_error = "no attempts made"
        attempts = 0
        for attempt in range(self.policy.max_retries + 1):
            attempts += 1
            try:
                output = _call_with_timeout(lambda: tool.fn(args), self.policy.timeout_s)
                return ToolResult(tool=tool.name, ok=True, output=output, attempts=attempts)
            except ToolInputError as exc:
                return ToolResult(tool=tool.name, ok=False, error=str(exc), attempts=attempts)
            except Exception as exc:  # any tool crash is a recorded failure, never a raise
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.policy.max_retries:
                self._sleep(self.policy.delay(attempt))
        return ToolResult(tool=tool.name, ok=False, error=last_error, attempts=attempts)
