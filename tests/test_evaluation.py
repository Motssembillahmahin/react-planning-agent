import json
from pathlib import Path

import pytest

from react_agent import evaluation as ev
from react_agent.types import AgentResult, GuardName, Status

TASKS = Path(__file__).resolve().parents[1] / "evals" / "tasks.yaml"


def test_tasks_file_is_valid_and_large_enough() -> None:
    tasks = ev.load_tasks(TASKS)
    assert len(tasks) >= 30
    for category in ev.CATEGORIES:
        assert sum(t.category == category for t in tasks) >= 5
    for task in tasks:
        ev.build_registry(task)  # every tool spec must build


def test_load_tasks_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "t.yaml"
    task = {
        "id": "a",
        "category": "solvable",
        "task": "t",
        "tools": [],
        "expected": {"status": ["success"]},
    }
    path.write_text(json.dumps({"tasks": [task, task]}))
    with pytest.raises(ValueError, match="duplicate"):
        ev.load_tasks(path)


def test_build_tool_faults_and_unknown_base() -> None:
    for fault in ("none", "flaky", "always_fail", "slow", "misleading"):
        tool = ev.build_tool(ev.ToolSpec(name="calc", base="calculator", fault=fault))
        assert tool.name == "calc"
    with pytest.raises(ValueError, match="unknown base tool"):
        ev.build_tool(ev.ToolSpec(name="weather"))


@pytest.mark.parametrize(
    ("answer", "expected", "ok"),
    [
        ("It is 395.", "395", True),
        ("about 10.69 times", "10.69", True),
        ("1,642 metres", "1642", True),
        ("394", "395", False),
        (None, "395", False),
        ("The city is Lisbon.", "Lisbon", True),
        ("Porto", "Lisbon", False),
    ],
)
def test_answer_matches(answer: str | None, expected: str, ok: bool) -> None:
    assert ev.answer_matches(answer, expected) is ok


def _result(
    status: Status, answer: str | None = None, guard: GuardName | None = None
) -> AgentResult:
    return AgentResult(status=status, answer=answer, guard=guard, reason="r", iterations=1)


def test_evaluate_outcome_rules() -> None:
    task = ev.TaskSpec(
        id="t", category="tool-failure", task="x", tools=[],
        expected=ev.Expected(status=[Status.SUCCESS, Status.ABSTAINED], answer="212"),
    )  # fmt: skip
    assert ev.evaluate_outcome(task, "v", _result(Status.SUCCESS, "212 F")).correct
    assert not ev.evaluate_outcome(task, "v", _result(Status.SUCCESS, "233 F")).correct
    assert ev.evaluate_outcome(task, "v", _result(Status.ABSTAINED)).correct
    capped = ev.evaluate_outcome(task, "v", _result(Status.ABSTAINED, guard=GuardName.SAFETY_CAP))
    assert not capped.terminated
    assert not capped.correct


def test_run_eval_summary_and_markdown(tmp_path: Path) -> None:
    tasks = ev.load_tasks(TASKS)
    outcomes = ev.run_eval(tasks, {"naive": ev.VARIANTS["naive"]}, trace_dir=tmp_path / "tr")
    summary = ev.summarize(outcomes)
    naive = summary["naive"]
    assert naive["tasks"] == len(tasks)
    assert naive["safety_cap_hits_on_loop_tasks"] >= 1
    assert (tmp_path / "tr" / "naive" / "solv-arith.jsonl").exists()
    ev.write_results(tmp_path, summary, outcomes, "tasks.yaml")
    md = (tmp_path / "results.md").read_text()
    assert "| naive |" in md
    assert "## Per-task outcomes" in md
    data = json.loads((tmp_path / "results.json").read_text())
    assert data["summary"]["naive"]["tasks"] == len(tasks)


def test_rate_and_pct_helpers() -> None:
    assert ev._rate(1, 0) is None
    assert ev._pct(None) == "n/a"
    assert ev._pct(0.5) == "50.0%"


def test_main_writes_results_and_rejects_unknown_variant(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = ev.main(
        [
            "--tasks",
            str(TASKS),
            "--out",
            str(tmp_path),
            "--trace-dir",
            str(tmp_path / "t"),
            "--variants",
            "naive",
        ]
    )
    assert rc == 0
    assert (tmp_path / "results.md").exists()
    assert "Evaluation results" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        ev.main(["--tasks", str(TASKS), "--variants", "bogus"])


def test_headline_claims_hold_for_all_variants() -> None:
    tasks = ev.load_tasks(TASKS)
    summary = ev.summarize(ev.run_eval(tasks, ev.VARIANTS))
    assert list(summary) == [
        "full",
        "no_critic",
        "no_repeat_detector",
        "no_progress_detector",
        "naive",
    ]
    for guarded in ("full", "no_critic", "no_repeat_detector", "no_progress_detector"):
        assert summary[guarded]["termination_rate"] == 1.0
        assert summary[guarded]["safety_cap_hits_on_loop_tasks"] == 0
    assert summary["naive"]["safety_cap_hits_on_loop_tasks"] >= summary["naive"]["loop_tasks"] - 1
    assert summary["full"]["success_rate"] > summary["no_critic"]["success_rate"]
    assert summary["full"]["success_rate"] > summary["naive"]["success_rate"]
    assert summary["full"]["mean_iterations"] < summary["no_progress_detector"]["mean_iterations"]


def test_main_live_provider_without_key_is_skipped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    assert ev.main(["--tasks", str(TASKS), "--out", str(tmp_path), "--provider", "auto"]) == 0
    assert "eval-real skipped: set ANTHROPIC_API_KEY" in capsys.readouterr().out
    assert ev.main(["--tasks", str(TASKS), "--out", str(tmp_path), "--provider", "anthropic"]) == 0
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().out
    assert not (tmp_path / "results.md").exists()


def test_main_live_provider_with_fake_llm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from react_agent.llm import ScriptedLLM

    def fake_make_llm(provider: str, model: str | None = None) -> ScriptedLLM:
        llm = ScriptedLLM([{"abstain": "offline fake"}], on_exhausted="repeat_last")
        llm.name = "fake"
        return llm

    monkeypatch.setattr(ev, "make_llm", fake_make_llm)
    rc = ev.main(["--tasks", str(TASKS), "--out", str(tmp_path), "--trace-dir", str(tmp_path / "t"),
                  "--provider", "openai", "--variants", "full"])  # fmt: skip
    assert rc == 0
    assert "live provider" in (tmp_path / "results.md").read_text()
