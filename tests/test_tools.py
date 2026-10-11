import math
import time

import pytest
from pydantic import BaseModel

from react_agent.tools import (
    FactStore,
    RetryPolicy,
    Tool,
    ToolExecutor,
    ToolInputError,
    ToolRegistry,
    always_fail,
    calculator_tool,
    default_registry,
    fact_store_tool,
    flaky,
    misleading,
    search_tool,
    slow,
    unit_converter_tool,
)
from react_agent.tools.base import NoArgs
from react_agent.tools.calculator import evaluate, format_number
from react_agent.tools.search import load_corpus, search, tokenize
from react_agent.tools.units import convert
from react_agent.types import Action

NO_WAIT = RetryPolicy(max_retries=2, backoff_s=0.0, timeout_s=None)


def run(registry: ToolRegistry, tool: str, policy: RetryPolicy = NO_WAIT, **args: object):
    return ToolExecutor(registry, policy).execute(Action(tool=tool, args=args))


# --- calculator -------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("17 * 23 + 4", 395),
        ("2 ^ 10", 1024),
        ("-(3 + 4) * 2", -14),
        ("7 // 2 + 7 % 2", 4),
        ("sqrt(16) + abs(-2)", 6.0),
        ("max(1, 5, 3) - min(4, 2)", 3),
        ("round(pi, 2)", 3.14),
        ("10 / 4", 2.5),
        ("+e", math.e),
    ],
)
def test_calculator_evaluates(expr: str, expected: float) -> None:
    assert evaluate(expr) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('ls')",
        "open('x')",
        "x + 1",
        "1 / 0",
        "2 ** 100000",
        "10 ** 999 * 10 ** 999 * 10 ** 999 * 10 ** 999 * 10 ** 999",
        "'a' * 3",
        "True + 1",
        "sqrt(-1)",
        "1 +",
        "1" * 201,
        "[1, 2]",
        "round(1.5, ndigits=1)",
        "exp(1000)",
    ],
)
def test_calculator_rejects_unsafe_or_invalid(expr: str) -> None:
    with pytest.raises(ToolInputError):
        evaluate(expr)


def test_format_number() -> None:
    assert format_number(4.0) == "4"
    assert format_number(1 / 3) == "0.3333333333"
    assert format_number(7) == "7"
    assert format_number(1e20) == "1e+20"


def test_calculator_tool_signature_and_run() -> None:
    reg = ToolRegistry([calculator_tool()])
    assert "calculator(expression: str)" in reg.describe()
    assert run(reg, "calculator", expression="2*21").output == "42"


# --- units -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "src", "dst", "expected"),
    [
        (100, "celsius", "fahrenheit", 212),
        (32, "F", "C", 0),
        (0, "c", "k", 273.15),
        (300, "K", "F", 80.33),
        (1, "mile", "km", 1.609344),
        (5, "kg", "lb", 11.0231),
        (2, "hours", "min", 120),
        (1, "gallon", "l", 3.7854),
        (100, "km/h", "m/s", 27.7778),
    ],
)
def test_unit_conversions(value: float, src: str, dst: str, expected: float) -> None:
    assert convert(value, src, dst) == pytest.approx(expected, rel=1e-3)


def test_unit_errors() -> None:
    with pytest.raises(ToolInputError, match="unknown unit"):
        convert(1, "parsec", "m")
    with pytest.raises(ToolInputError, match="cannot convert"):
        convert(1, "kg", "m")


def test_unit_tool_output() -> None:
    reg = ToolRegistry([unit_converter_tool()])
    res = run(reg, "unit_converter", value=100, from_unit="C", to_unit="F")
    assert res.output == "100 C = 212 F"


# --- search ------------------------------------------------------------------


def test_search_finds_relevant_doc() -> None:
    hits = search("Eiffel Tower height")
    assert hits[0].id == "eiffel"
    assert len(load_corpus()) >= 20


def test_search_no_results_and_stopwords() -> None:
    assert search("the of and") == []
    assert search("zxqv") == []
    assert tokenize("What is The Height?") == ["height"]


def test_search_tool_output() -> None:
    reg = ToolRegistry([search_tool()])
    assert "[everest]" in run(reg, "search", query="Mount Everest elevation").output
    assert run(reg, "search", query="unicorn").output == "No results for 'unicorn'."


# --- facts -------------------------------------------------------------------


def test_fact_store_ops() -> None:
    store = FactStore()
    reg = ToolRegistry([fact_store_tool(store)])
    assert run(reg, "facts", op="get", key="Team_Size").output == "team_size = 12"
    assert "No fact stored" in run(reg, "facts", op="get", key="ceo").output
    assert run(reg, "facts", op="set", key="ceo", value="Ana").output == "stored ceo = Ana"
    assert store.facts["ceo"] == "Ana"
    assert "office_city" in run(reg, "facts", op="list").output
    assert run(reg, "facts", op="get").error == "'get' requires a key"
    assert run(reg, "facts", op="set", key="k").error == "'set' requires a value"
    empty = ToolRegistry([fact_store_tool(FactStore({}))])
    assert run(empty, "facts", op="list").output == "keys: (empty)"


# --- executor ----------------------------------------------------------------


class _Counter:
    def __init__(self, failures: int, exc: Exception) -> None:
        self.failures = failures
        self.exc = exc
        self.calls = 0

    def __call__(self, args: BaseModel) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return "ok"


def _tool(name: str, fn: object) -> Tool:
    return Tool(name=name, description="test tool", args_model=NoArgs, fn=fn)  # type: ignore[arg-type]


def test_unknown_tool_and_invalid_args() -> None:
    reg = default_registry()
    res = run(reg, "nope")
    assert not res.ok
    assert "unknown tool 'nope'" in (res.error or "")
    assert res.attempts == 0
    assert "unknown tool" in (run(ToolRegistry(), "x").error or "")
    bad = run(reg, "calculator", wrong=1)
    assert not bad.ok
    assert "invalid args" in (bad.error or "")


def test_retry_with_backoff_then_success() -> None:
    sleeps: list[float] = []
    counter = _Counter(2, RuntimeError("boom"))
    reg = ToolRegistry([_tool("t", counter)])
    policy = RetryPolicy(max_retries=2, backoff_s=0.1, backoff_factor=2.0, timeout_s=None)
    res = ToolExecutor(reg, policy, sleep=sleeps.append).execute(Action(tool="t"))
    assert res.ok
    assert res.attempts == 3
    assert sleeps == [0.1, 0.2]


def test_retries_exhausted_records_failure() -> None:
    reg = ToolRegistry([_tool("t", _Counter(99, RuntimeError("down")))])
    res = run(reg, "t")
    assert not res.ok
    assert res.attempts == 3
    assert res.error == "RuntimeError: down"
    assert res.as_observation() == "ERROR from t: RuntimeError: down"


def test_input_errors_are_not_retried() -> None:
    counter = _Counter(99, ToolInputError("bad"))
    res = run(ToolRegistry([_tool("t", counter)]), "t")
    assert not res.ok
    assert counter.calls == 1


def test_fallback_used_after_retries() -> None:
    reg = ToolRegistry()
    reg.register(always_fail(calculator_tool()), fallback="calc_backup")
    reg.register(calculator_tool("calc_backup"))
    res = run(reg, "calculator", expression="6*7")
    assert res.ok
    assert res.output == "42"
    assert res.used_fallback == "calc_backup"
    assert res.tool == "calculator"
    assert res.attempts == 4
    assert res.as_observation() == "[calc_backup (fallback)] 42"


def test_fallback_failure_and_arg_mismatch() -> None:
    reg = ToolRegistry([always_fail(calculator_tool()), always_fail(calculator_tool("b"))])
    reg.set_fallback("calculator", "b")
    res = run(reg, "calculator", expression="1")
    assert not res.ok
    assert "fallback b failed" in (res.error or "")
    reg2 = ToolRegistry([always_fail(calculator_tool()), search_tool()])
    reg2.set_fallback("calculator", "search")
    res2 = run(reg2, "calculator", expression="1")
    assert "fallback rejected args" in (res2.error or "")
    reg3 = ToolRegistry([always_fail(calculator_tool())])
    reg3.set_fallback("calculator", "missing")
    assert not run(reg3, "calculator", expression="1").ok


def test_timeout_is_a_failure() -> None:
    reg = ToolRegistry([slow(calculator_tool(), delay_s=0.3)])
    policy = RetryPolicy(max_retries=0, backoff_s=0.0, timeout_s=0.05)
    start = time.monotonic()
    res = run(reg, "calculator", policy, expression="1+1")
    assert time.monotonic() - start < 0.25
    assert not res.ok
    assert "timed out" in (res.error or "")


def test_slow_within_timeout_succeeds() -> None:
    slept: list[float] = []
    reg = ToolRegistry([slow(calculator_tool(), delay_s=5, sleep=slept.append)])
    res = run(reg, "calculator", RetryPolicy(timeout_s=1.0), expression="1+1")
    assert res.ok
    assert slept == [5]


# --- faults ------------------------------------------------------------------


def test_flaky_is_seeded_and_reproducible() -> None:
    def outcomes(seed: int) -> list[bool]:
        reg = ToolRegistry([flaky(calculator_tool(), p=0.5, seed=seed)])
        policy = RetryPolicy(max_retries=0, backoff_s=0, timeout_s=None)
        return [run(reg, "calculator", policy, expression="1").ok for _ in range(20)]

    assert outcomes(3) == outcomes(3)
    assert True in outcomes(3)
    assert False in outcomes(3)
    always = ToolRegistry([flaky(calculator_tool(), p=0.0)])
    assert run(always, "calculator", expression="2").ok


def test_misleading_perturbs_numbers() -> None:
    reg = ToolRegistry([misleading(search_tool())])
    out = run(reg, "search", query="Eiffel Tower").output
    assert "330" not in out
    assert "363" in out
    calc = ToolRegistry([misleading(calculator_tool())])
    assert run(calc, "calculator", expression="2.5*1").output == "2.8"
    assert run(calc, "calculator", expression="3").output == "4"
    assert misleading(search_tool()).metadata["fault"] == "misleading"
