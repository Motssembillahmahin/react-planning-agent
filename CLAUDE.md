# CLAUDE.md — conventions for agents working in this repo

## Commands (the Makefile is the only interface)
- `make install`   — `uv sync` (dev group included)
- `make lint`      — `ruff check` + `ruff format --check`
- `make format`    — apply ruff formatting and autofixes
- `make typecheck` — `mypy --strict` on `src/react_agent`
- `make test`      — pytest + hypothesis, fails under 90% coverage
- `make eval`      — deterministic eval with ScriptedLLM → `evals/results.{md,json}`
- `make eval-real` — real provider eval, only if `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` is set (costs money; never run in CI)
- `make demo`      — scripted end-to-end run

## Layout
- `src/react_agent/` — the package (loop, guards, critic, tools, llm, cli, evaluation)
- `tests/` — unit + property tests (`ScriptedLLM` only, never network)
- `evals/tasks.yaml` — eval tasks; `evals/results.*` are generated, commit them after `make eval`
- `state.md` — handoff log; read it first, update it after every stage

## Conventions
- Python 3.12, pydantic v2 models, full type hints, `mypy --strict` clean.
- The agent never raises out of `run()`: every failure becomes an `AgentResult`.
- Every phase transition is written to the JSONL trace.
- Tests never call paid APIs. Provider classes are tested with fake clients.
- Keys come from environment variables only. Never commit `.env` or keys.
- Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`), one logical change per commit, push after each. Never force-push.
- Do not weaken a test or check to make it pass.
