"""The README must carry the real ablation table from evals/results.md, not a stale copy."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _ablation_rows(text: str) -> list[str]:
    return [
        line
        for line in text.splitlines()
        if line.startswith(("| variant", "| full", "| no_", "| naive"))
    ]


def test_readme_ablation_table_matches_results() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = readme.split("<!-- results:start -->")[1].split("<!-- results:end -->")[0]
    results = (ROOT / "evals" / "results.md").read_text(encoding="utf-8")
    ablation = results.split("## Ablation")[1].split("## Success by category")[0]
    assert _ablation_rows(block) == _ablation_rows(ablation)
    assert len(_ablation_rows(block)) == 6


def test_readme_has_required_sections() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for heading in ("## The problem", "```mermaid", "## Termination guarantees", "## Results",
                    "## Quickstart", "## Add a tool", "## Limitations"):  # fmt: skip
        assert heading in readme
    assert "TODO" not in readme
    assert "TBD" not in readme
