"""Fault-injection wrappers. Each keeps the wrapped tool's name and schema."""

from __future__ import annotations

import dataclasses
import random
import re
import time
from collections.abc import Callable
from typing import Any

from react_agent.tools.base import Tool, ToolError

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def flaky(tool: Tool, p: float, seed: int = 0) -> Tool:
    """Fails with probability `p`, using a seeded RNG so runs are reproducible."""
    rng = random.Random(seed)

    def fn(args: Any) -> str:
        if rng.random() < p:
            raise ToolError(f"transient failure in {tool.name}")
        return tool.fn(args)

    return dataclasses.replace(tool, fn=fn, metadata={**tool.metadata, "fault": "flaky"})


def always_fail(tool: Tool) -> Tool:
    def fn(args: Any) -> str:
        raise ToolError(f"{tool.name} is unavailable")

    return dataclasses.replace(tool, fn=fn, metadata={**tool.metadata, "fault": "always_fail"})


def slow(tool: Tool, delay_s: float, sleep: Callable[[float], None] = time.sleep) -> Tool:
    """Sleeps `delay_s` before answering, to exceed the executor timeout."""

    def fn(args: Any) -> str:
        sleep(delay_s)
        return tool.fn(args)

    return dataclasses.replace(tool, fn=fn, metadata={**tool.metadata, "fault": "slow"})


def _perturb(match: re.Match[str]) -> str:
    text = match.group(0)
    if "." in text:
        decimals = len(text.split(".")[1])
        return f"{float(text) * 1.1:.{decimals}f}"
    n = int(text)
    return str(n + max(1, n // 10))


def misleading(tool: Tool) -> Tool:
    """Returns plausible but wrong data: every number in the output is shifted by ~10%."""

    def fn(args: Any) -> str:
        return _NUMBER.sub(_perturb, tool.fn(args))

    return dataclasses.replace(tool, fn=fn, metadata={**tool.metadata, "fault": "misleading"})
