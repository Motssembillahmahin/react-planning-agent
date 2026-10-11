"""Tools available to the agent, plus fault-injection wrappers."""

from react_agent.tools.base import (
    RetryPolicy,
    Tool,
    ToolError,
    ToolExecutor,
    ToolInputError,
    ToolRegistry,
    ToolTimeoutError,
)
from react_agent.tools.calculator import calculator_tool
from react_agent.tools.facts import FactStore, fact_store_tool
from react_agent.tools.faults import always_fail, flaky, misleading, slow
from react_agent.tools.search import search_tool
from react_agent.tools.units import unit_converter_tool

__all__ = [
    "FactStore",
    "RetryPolicy",
    "Tool",
    "ToolError",
    "ToolExecutor",
    "ToolInputError",
    "ToolRegistry",
    "ToolTimeoutError",
    "always_fail",
    "calculator_tool",
    "default_registry",
    "fact_store_tool",
    "flaky",
    "misleading",
    "search_tool",
    "slow",
    "unit_converter_tool",
]


def default_registry() -> ToolRegistry:
    """The four real tools with no faults."""
    return ToolRegistry(
        [calculator_tool(), unit_converter_tool(), search_tool(), fact_store_tool(FactStore())]
    )
