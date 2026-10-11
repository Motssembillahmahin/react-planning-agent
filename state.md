# state.md — handoff log

## Task
Build, test, document, and publish `react-planning-agent`: a ReAct agent
(observe → think → act → reflect) with termination guards, self-critique, and
graceful degradation. Done = all acceptance checks in the original brief pass
and are pushed (see `checks.md` once written).

## Outputs (exact paths)
- Repo: https://github.com/Motssembillahmahin/react-planning-agent
- Local: /home/pulok/Documents/react-planning-agent
- Scaffolding: `pyproject.toml`, `uv.lock`, `Makefile`, `.python-version`
- Anchor docs: `CLAUDE.md`, `AGENTS.md`, `vision.md`, `state.md`
- Package: `src/react_agent/{types,trace,prompts,baseline,evaluation}.py`, `src/react_agent/llm/`, `src/react_agent/tools/`, `src/react_agent/data/corpus.json`
- Eval: `evals/tasks.yaml` (36 tasks), `evals/results.md`, `evals/results.json`
- Tests: `tests/test_llm_scripted.py`, `tests/test_tools.py`, `tests/test_baseline_trace.py`, `tests/test_evaluation.py`

## Completed stages + observed check results
1. Preflight — `gh auth status`: logged in as Motssembillahmahin (scopes repo, workflow).
   git identity Mahin <asmahin8@gmail.com> (set locally in repo; global is a different user).
   `uv --version`: 0.11.19. `gh repo view .../react-planning-agent`: not found → name free.
2. Initial commit (.gitignore, LICENSE, stub README) + `gh repo create --public --push`: ok.
3. Scaffolding: `uv sync` ok.
4. Core types + LLM interface + ScriptedLLM + parser: tests pass, ruff/mypy clean.
5. Tools (calculator, unit converter, search, fact store), executor (retry/backoff/timeout/fallback),
   fault wrappers (flaky, always_fail, slow, misleading): tests pass.
6. Tracer, prompts, NaiveReActAgent: tests pass.
7. Eval harness + 36 tasks (10 solvable, 8 unsolvable, 8 loop-inducing, 10 tool-failure).
   **Baseline numbers (naive, recorded before the guarded agent exists):**
   termination 80.6%, task success 44.4%, correct abstention 25.0%, mean iterations 11.69,
   mean tokens 11038.7, safety-cap hits on loop tasks 7/8 (the 8th, loop-same-answer, returns
   its first unsupported answer). Success by category: solvable 80%, unsolvable 25%,
   loop-inducing 0%, tool-failure 60%.
8. Guards (`src/react_agent/guards.py`, `src/react_agent/config.py`): max_iterations, token budget,
   timeout (injected clock), repeated-action (normalised fingerprint, sliding window),
   no-progress, tool error budget; each disabled by setting it to None. `tests/test_guards.py` passes.
9. Critic (`src/react_agent/critic.py`): EvidenceCritic (default), LLMCritic, CompositeCritic.
10. ReActAgent (`src/react_agent/agent.py`): explicit state machine; every transition traced.
    Property test (`tests/test_property.py`): 300 passing examples; a deliberate off-by-one in
    MaxIterationsGuard makes it fail (mutation check), then restored.
    `make lint typecheck test`: all pass, coverage 99.9%.
11. Ablation eval (`make eval`): full 100% termination / 94.4% success / 87.5% correct abstention /
    2.83 mean iterations; no_critic 72.2% success, 25% abstention; no_repeat_detector 94.4%
    (2.94 it.); no_progress_detector 94.4% (3.33 it.); naive 80.6% termination, 7/8 loop tasks hit
    the 50-step cap. Re-running gives byte-identical results.json.
12. Providers (`src/react_agent/llm/{anthropic_llm,openai_llm,providers}.py`): fake-client tests
    only; never called a live API. `make eval-real` / `--provider auto` print a skip message
    when no key is set (tested with env vars removed).
13. CLI (`src/react_agent/cli.py`): `make demo` → success after 1 critic rejection;
    `react-agent replay` renders the trace. `tests/test_cli.py` passes.
14. CI (`.github/workflows/ci.yml`): first run failed (`astral-sh/setup-uv@v10` has no floating
    tag) → pinned v10.3.0 → green (run 38113686142). Runner pinned to ubuntu-24.04.
15. README with all required sections; `tests/test_docs.py` keeps its ablation table identical
    to `evals/results.md`. README tool snippet executed end to end successfully.

## Decisions + reasons
- Repo name `react-planning-agent` (did not exist).
- Repo-local git identity, because global identity is a different user.
- `pyyaml` added as a runtime dep to load `evals/tasks.yaml`.
- Eval harness lives in the package (`react_agent.evaluation`) so it is typed,
  linted and covered; `make eval` calls it as a module.

- Correctness requires termination; expected answers are checked only on `success`.
- Scripts are fixed (the model does not react to observations), so the naive baseline can be
  "lucky" on recoverable tool failures (it answers correctly despite seeing an error). Kept as is
  and listed as a limitation rather than tuning scripts to flatter the guarded agent.
- Two tasks deliberately expose critic blind spots (unsolv-neptune-sqrt, fail-misleading-search).
- Status mapping: model abstain / critic cap → abstained; any guard with a candidate answer →
  partial; repeated-action / no-progress without one → abstained; resource guards → failed.
- Reflection is deterministic: a step adds information iff a successful observation contains
  content words/numbers not seen before (task + earlier observations), excluding echoes of the
  action's own args. Not LLM-based, so it is cheap and testable.
- Critic rejections do not count toward the no-progress streak; the rejection cap bounds them.
- llm_calls field dropped from AgentResult (ambiguous with an LLM critic); bound is asserted on
  ScriptedLLM.calls in the property test instead.
- AnthropicLLM default model `claude-opus-5` with server-side refusal fallbacks on by default
  (beta `server-side-fallback-2026-07-01`, `fallbacks="default"`), per the claude-api reference;
  `refusal_fallbacks=False` opts out. OpenAI-compatible default `gpt-4o-mini`; Ollama `llama3.1`.
- CI also runs `git diff --exit-code` on eval results to prove determinism.
- Flaky seeds chosen by enumeration: p=0.5 seed 7 = fail, fail, ok; p=0.7 seed 4 = fail x3.

## Open issues
- Repeated-action and no-progress detectors overlap on this task set (either alone catches every
  loop); the ablation shows only an iteration/token cost. Reported honestly in the README.

## Next action
Run the acceptance checks on a clean clone and write `checks.md`.
