import pytest

from react_agent.config import AgentConfig
from react_agent.guards import (
    Checkpoint,
    GuardSet,
    LoopState,
    MaxIterationsGuard,
    NoProgressGuard,
    RepeatedActionGuard,
    TimeoutGuard,
    TokenBudgetGuard,
    ToolErrorBudgetGuard,
    action_fingerprint,
    build_guards,
)
from react_agent.types import Action, GuardName


def state(**kw: int) -> LoopState:
    return LoopState(started_at=0.0, **kw)


def test_max_iterations() -> None:
    g = MaxIterationsGuard(3)
    assert g.check(state(iterations=2), None) is None
    trip = g.check(state(iterations=3), None)
    assert trip is not None
    assert trip.guard is GuardName.MAX_ITERATIONS


def test_token_budget() -> None:
    g = TokenBudgetGuard(100)
    assert g.check(state(tokens_used=99), None) is None
    assert g.check(state(tokens_used=100), None) is not None


def test_timeout_uses_injected_clock() -> None:
    now = [5.0]
    g = TimeoutGuard(10.0, clock=lambda: now[0])
    assert g.check(state(), None) is None
    now[0] = 10.0
    trip = g.check(state(), None)
    assert trip is not None
    assert "limit 10s" in trip.reason


def test_fingerprint_normalisation() -> None:
    a = Action(tool="Unit_Converter", args={"value": 5, "from_unit": "km", "to_unit": "mi"})
    b = Action(tool="unit_converter ", args={"to_unit": " MI", "value": 5.0, "FROM_UNIT": "Km"})
    c = Action(tool="unit_converter", args={"value": 6, "from_unit": "km", "to_unit": "mi"})
    assert action_fingerprint(a) == action_fingerprint(b)
    assert action_fingerprint(a) != action_fingerprint(c)
    nested = Action(tool="t", args={"xs": [1.0, "A  b"], "flag": True, "n": None})
    assert action_fingerprint(nested) == action_fingerprint(
        Action(tool="t", args={"xs": [1, "a b"], "flag": True, "n": None})
    )
    assert action_fingerprint(Action(tool="t", args={"f": True})) != action_fingerprint(
        Action(tool="t", args={"f": 1})
    )


def test_repeated_action_sliding_window() -> None:
    act = Action(tool="search", args={"query": "x"})
    other = Action(tool="search", args={"query": "y"})
    fp, fo = action_fingerprint(act), action_fingerprint(other)
    g = RepeatedActionGuard(window=4, threshold=3)
    assert g.check(state(), None) is None
    assert g.check(LoopState(0.0, action_history=[fp]), act) is None
    trip = g.check(LoopState(0.0, action_history=[fp, fo, fp]), act)
    assert trip is not None
    assert trip.guard is GuardName.REPEATED_ACTION
    # An old occurrence that slid out of the window does not count.
    assert g.check(LoopState(0.0, action_history=[fp, fo, fo, fp]), act) is None
    single = RepeatedActionGuard(window=1, threshold=2)
    assert single.check(LoopState(0.0, action_history=[fp]), act) is None


def test_no_progress() -> None:
    g = NoProgressGuard(2)
    assert g.check(state(no_progress_streak=1), None) is None
    assert g.check(state(no_progress_streak=2), None) is not None


def test_tool_error_budget() -> None:
    g = ToolErrorBudgetGuard(3)
    assert g.check(state(tool_errors=2), None) is None
    assert g.check(state(tool_errors=3), None) is not None


def test_guardset_only_checks_matching_checkpoint() -> None:
    gs = GuardSet([MaxIterationsGuard(1), ToolErrorBudgetGuard(1)])
    s = state(iterations=5, tool_errors=5)
    trip = gs.check(Checkpoint.BEFORE_THINK, s)
    assert trip is not None
    assert trip.guard is GuardName.MAX_ITERATIONS
    trip2 = gs.check(Checkpoint.AFTER_ACT, s)
    assert trip2 is not None
    assert trip2.guard is GuardName.TOOL_ERROR_BUDGET
    assert gs.check(Checkpoint.AFTER_REFLECT, s) is None


def test_build_guards_respects_disabled() -> None:
    full = build_guards(AgentConfig())
    assert full.names() == [
        "max_iterations", "token_budget", "timeout", "repeated_action", "no_progress",
        "tool_error_budget",
    ]  # fmt: skip
    minimal = build_guards(
        AgentConfig(
            token_budget=None,
            timeout_s=None,
            repeat_threshold=None,
            no_progress_patience=None,
            tool_error_budget=None,
        )
    )
    assert minimal.names() == ["max_iterations"]


def test_config_validation_and_retry_policy() -> None:
    with pytest.raises(ValueError, match="greater than or equal"):
        AgentConfig(max_iterations=0)
    policy = AgentConfig(tool_max_retries=1, tool_backoff_s=0.5, tool_timeout_s=None).retry_policy()
    assert (policy.max_retries, policy.backoff_s, policy.timeout_s) == (1, 0.5, None)
