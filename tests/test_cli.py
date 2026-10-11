import json
from pathlib import Path

import pytest
from rich.console import Console

from react_agent import cli

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo_script.json"


def console() -> Console:
    return Console(record=True, width=200)


def test_run_scripted_success_and_replay(tmp_path: Path) -> None:
    c = console()
    rc = cli.main(
        ["run", "What is 17 * 23 + 4?", "--script", str(DEMO), "--trace-dir", str(tmp_path)], c
    )
    assert rc == 0
    out = c.export_text()
    assert "success" in out
    assert "395" in out
    (trace,) = tmp_path.glob("run-*.jsonl")
    c2 = console()
    assert cli.main(["replay", str(trace)], c2) == 0
    replay = c2.export_text()
    assert "THINK → CRITIQUE" in replay
    assert "CRITIQUE → DONE" in replay


def test_run_json_output_and_failure_exit_code(tmp_path: Path) -> None:
    script = tmp_path / "s.json"
    script.write_text(json.dumps([{"final_answer": "made up"}]))
    c = console()
    rc = cli.main(
        [
            "run",
            "q",
            "--script",
            str(script),
            "--trace-dir",
            str(tmp_path),
            "--json",
            "--max-iterations",
            "3",
        ],
        c,
    )
    assert rc == 1
    assert '"status": "partial"' in c.export_text()
    c2 = console()
    rc2 = cli.main(
        ["run", "q", "--script", str(script), "--trace-dir", str(tmp_path), "--no-critic"], c2
    )
    assert rc2 == 0


def test_run_without_provider_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    c = console()
    assert cli.main(["run", "q", "--trace-dir", str(tmp_path)], c) == 2
    assert "cannot start" in c.export_text()
    c2 = console()
    assert cli.main(["run", "q", "--script", str(tmp_path / "missing.json")], c2) == 2


def test_replay_errors_and_truncation(tmp_path: Path) -> None:
    c = console()
    assert cli.main(["replay", str(tmp_path / "nope.jsonl")], c) == 2
    assert "cannot read trace" in c.export_text()
    trace = tmp_path / "t.jsonl"
    trace.write_text(
        json.dumps({"seq": 0, "ts": 0, "run_id": "r", "event": "note", "text": "x" * 500}) + "\n"
    )
    c2 = console()
    assert cli.main(["replay", str(trace)], c2) == 0
    assert "..." in c2.export_text()
