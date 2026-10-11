# Acceptance checks

Verified on 2026-10-11 at commit `acc5f15` (the last commit before this file).

| requirement | verdict | evidence | correction made |
|---|---|---|---|
| 1. Public GitHub repo exists; every commit pushed to `main` | pass | `gh repo view --json visibility` → `PUBLIC https://github.com/Motssembillahmahin/react-planning-agent`; `git rev-parse HEAD origin/main` → both `acc5f15fdb2e61b2698538c8f0ffc64bd816b0f9`; `git status -sb` → `## main...origin/main` | Second push failed (`could not read Username`): added a repo-local gh credential helper and re-pushed. No force-push, no history rewrite. |
| 2. `make install lint typecheck test eval` passes on a clean clone | pass | Fresh `git clone` into a scratch dir: `make exit=0`; ruff "All checks passed!", "44 files already formatted"; mypy "Success: no issues found in 27 source files"; "153 passed"; `git status --porcelain` empty after `make eval` (results reproduce byte-for-byte) | none |
| 3. Coverage ≥ 90% for `src/react_agent/` | pass | Clean clone: "Required test coverage of 90% reached. Total coverage: 99.93%" (`--cov-fail-under=90` enforced in `make test` and CI) | none |
| 4. Property test exists and passes with ≥ 200 examples | pass | `tests/test_property.py`, `max_examples=300`; `pytest --hypothesis-show-statistics` → "300 passing, 0 failing". Mutation check: making `MaxIterationsGuard` off by one makes it fail (`assert 2 <= 1`). | none |
| 5. `evals/results.md`: full agent 100% termination; naive baseline hits the safety cap on loop-inducing tasks | pass | `evals/results.md` ablation row `full | 100.0% | ... | 0/8`; `naive | 80.6% | ... | 7/8` (7 of 8 loop tasks end with `failed / safety_cap`; the 8th, `loop-same-answer`, returns an unsupported answer). `tests/test_evaluation.py::test_headline_claims_hold_for_all_variants` guards this. | none |
| 6. Latest GitHub Actions CI run is green | pass | https://github.com/Motssembillahmahin/react-planning-agent/actions/runs/38113816274 (`completed success`, head `acc5f15`). The run for this file's own commit is reported in the final summary. | First CI run failed: `astral-sh/setup-uv@v10` has no floating tag → pinned `v10.3.0`; runner pinned to `ubuntu-24.04` ahead of the `ubuntu-latest` migration notice. |
| 7. README has every required section with real numbers | pass | `README.md`: problem statement, Mermaid `stateDiagram-v2`, termination-guarantee table, ablation table (identical to `evals/results.md`, enforced by `tests/test_docs.py`), Makefile quickstart, "Add a tool" (snippet executed end to end: `status=success, answer='3 words'`), Limitations. No TODO/TBD (tested). | The first draft said the baseline is "lucky" on 5 tasks; checking the traces showed 3 lucky + 1 ignored-evidence + 2 genuine (slow tools). Corrected before commit. |
| 8. No secrets: `git log -p \| grep -iE "sk-\|api_key="` returns only code that reads env vars | pass | Output is exactly 2 lines: `+export ANTHROPIC_API_KEY=...` (README instructions, placeholder) and `+ client = sdk.OpenAI(api_key=api_key or "local", ...)` where `api_key = os.environ.get("OPENAI_API_KEY")`. Tests use `test-not-a-real-key` via `monkeypatch.setenv`. `.env*` is gitignored. | none |

## Unresolved

- No live-provider numbers: `make eval-real` was deliberately never run (no paid API calls).
  The provider classes are covered only by fake-client tests.
