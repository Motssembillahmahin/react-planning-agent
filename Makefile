.PHONY: install lint format typecheck test eval eval-real demo

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy src/react_agent

test:
	uv run pytest --cov=react_agent --cov-report=term-missing --cov-fail-under=90

eval:
	uv run python -m react_agent.evaluation --tasks evals/tasks.yaml --out evals

# Uses a real LLM provider only if ANTHROPIC_API_KEY or OPENAI_API_KEY is set.
eval-real:
	uv run --extra anthropic --extra openai python -m react_agent.evaluation \
		--tasks evals/tasks.yaml --out evals/real --provider auto

demo:
	uv run react-agent run "What is 17 * 23 + 4?" --script examples/demo_script.json
