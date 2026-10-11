"""Evaluation harness: run every task under every agent variant and report metrics.

Usage: python -m react_agent.evaluation --tasks evals/tasks.yaml --out evals
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from pydantic import BaseModel, Field

from react_agent.agent import ReActAgent
from react_agent.baseline import NaiveReActAgent
from react_agent.config import AgentConfig
from react_agent.llm.base import LLM, LLMError
from react_agent.llm.providers import available_provider, make_llm
from react_agent.llm.scripted import OnExhausted, ScriptedLLM
from react_agent.tools import (
    FactStore,
    Tool,
    ToolRegistry,
    always_fail,
    calculator_tool,
    fact_store_tool,
    flaky,
    misleading,
    search_tool,
    slow,
    unit_converter_tool,
)
from react_agent.types import AgentResult, GuardName, Status

Category = Literal["solvable", "unsolvable", "loop-inducing", "tool-failure"]
CATEGORIES: tuple[Category, ...] = ("solvable", "unsolvable", "loop-inducing", "tool-failure")
BaseTool = Literal["calculator", "unit_converter", "search", "facts"]
Fault = Literal["none", "flaky", "always_fail", "slow", "misleading"]

# Guarded variants use a 0.2s tool timeout in the eval; slow tools sleep longer than this.
EVAL_CONFIG = AgentConfig(tool_backoff_s=0.0, tool_timeout_s=0.2)


class ToolSpec(BaseModel):
    name: str
    base: BaseTool | None = None
    fault: Fault = "none"
    p: float = Field(default=0.5, ge=0.0, le=1.0)
    seed: int = 0
    delay_s: float = 0.5
    fallback: str | None = None


class Expected(BaseModel):
    status: list[Status]
    answer: str | None = None


class TaskSpec(BaseModel):
    id: str
    category: Category
    task: str
    tools: list[ToolSpec | str]
    script: list[dict[str, Any] | str] = Field(default_factory=list)
    on_exhausted: OnExhausted = "error"
    expected: Expected
    note: str = ""


def load_tasks(path: Path) -> list[TaskSpec]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    tasks = [TaskSpec.model_validate(t) for t in raw["tasks"]]
    ids = [t.id for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task ids")
    return tasks


_BASE_FACTORIES: dict[str, Callable[[str], Tool]] = {
    "calculator": calculator_tool,
    "unit_converter": unit_converter_tool,
    "search": search_tool,
    "facts": lambda name: fact_store_tool(FactStore(), name=name),
}


def build_tool(spec: ToolSpec) -> Tool:
    base = spec.base or spec.name
    if base not in _BASE_FACTORIES:
        raise ValueError(f"unknown base tool '{base}'")
    tool = _BASE_FACTORIES[base](spec.name)
    if spec.fault == "flaky":
        return flaky(tool, spec.p, spec.seed)
    if spec.fault == "always_fail":
        return always_fail(tool)
    if spec.fault == "slow":
        return slow(tool, spec.delay_s)
    if spec.fault == "misleading":
        return misleading(tool)
    return tool


def build_registry(task: TaskSpec) -> ToolRegistry:
    registry = ToolRegistry()
    for item in task.tools:
        spec = ToolSpec(name=item) if isinstance(item, str) else item
        registry.register(build_tool(spec), fallback=spec.fallback)
    return registry


class Runner(Protocol):
    def run(self, task: str) -> AgentResult: ...


AgentFactory = Callable[[LLM, ToolRegistry, Path | None], Runner]


def _naive(llm: LLM, tools: ToolRegistry, trace: Path | None) -> Runner:
    return NaiveReActAgent(llm, tools, trace_path=trace)


def _guarded(**overrides: Any) -> AgentFactory:
    config = EVAL_CONFIG.model_copy(update=overrides)

    def factory(llm: LLM, tools: ToolRegistry, trace: Path | None) -> Runner:
        return ReActAgent(llm, tools, config, trace_path=trace)

    return factory


VARIANTS: dict[str, AgentFactory] = {
    "full": _guarded(),
    "no_critic": _guarded(critic_enabled=False),
    "no_repeat_detector": _guarded(repeat_threshold=None),
    "no_progress_detector": _guarded(no_progress_patience=None),
    "naive": _naive,
}


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in _NUM.findall(text)]


def answer_matches(answer: str | None, expected: str) -> bool:
    """Numeric expectations match any number in the answer (rel. tol 1e-4); text by substring."""
    if answer is None:
        return False
    exp_nums = _numbers(expected)
    if exp_nums and re.fullmatch(r"\s*" + _NUM.pattern + r"\s*", expected):
        target = exp_nums[0]
        return any(abs(n - target) <= 1e-4 * max(1.0, abs(target)) for n in _numbers(answer))
    return expected.strip().lower() in answer.lower()


class Outcome(BaseModel):
    task_id: str
    category: Category
    variant: str
    status: Status
    answer: str | None
    guard: GuardName | None
    iterations: int
    tokens: int
    correct: bool
    terminated: bool


def evaluate_outcome(task: TaskSpec, variant: str, result: AgentResult) -> Outcome:
    terminated = result.guard not in (GuardName.SAFETY_CAP, GuardName.INTERNAL_ERROR)
    answer_ok = (
        result.status is not Status.SUCCESS
        or task.expected.answer is None
        or answer_matches(result.answer, task.expected.answer)
    )
    correct = terminated and result.status in task.expected.status and answer_ok
    return Outcome(
        task_id=task.id,
        category=task.category,
        variant=variant,
        status=result.status,
        answer=result.answer,
        guard=result.guard,
        iterations=result.iterations,
        tokens=result.tokens_used,
        correct=correct,
        terminated=terminated,
    )


LLMFactory = Callable[[TaskSpec], LLM]


def scripted_llm(task: TaskSpec) -> LLM:
    return ScriptedLLM(task.script, on_exhausted=task.on_exhausted)


def run_eval(
    tasks: list[TaskSpec],
    variants: dict[str, AgentFactory],
    *,
    llm_factory: LLMFactory = scripted_llm,
    trace_dir: Path | None = None,
) -> list[Outcome]:
    outcomes: list[Outcome] = []
    for name, factory in variants.items():
        for task in tasks:
            trace = trace_dir / name / f"{task.id}.jsonl" if trace_dir else None
            agent = factory(llm_factory(task), build_registry(task), trace)
            outcomes.append(evaluate_outcome(task, name, agent.run(task.task)))
    return outcomes


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def summarize(outcomes: list[Outcome]) -> dict[str, Any]:
    """Aggregate metrics per variant."""
    summary: dict[str, Any] = {}
    for variant in dict.fromkeys(o.variant for o in outcomes):
        rows = [o for o in outcomes if o.variant == variant]
        unsolvable = [o for o in rows if o.category == "unsolvable"]
        loops = [o for o in rows if o.category == "loop-inducing"]
        n = len(rows)
        summary[variant] = {
            "tasks": n,
            "termination_rate": _rate(sum(o.terminated for o in rows), n),
            "success_rate": _rate(sum(o.correct for o in rows), n),
            "correct_abstention_rate": _rate(
                sum(o.status is Status.ABSTAINED for o in unsolvable), len(unsolvable)
            ),
            "mean_iterations": round(sum(o.iterations for o in rows) / n, 2),
            "mean_tokens": round(sum(o.tokens for o in rows) / n, 1),
            "safety_cap_hits_on_loop_tasks": sum(o.guard is GuardName.SAFETY_CAP for o in loops),
            "loop_tasks": len(loops),
            "success_by_category": {
                c: _rate(
                    sum(o.correct for o in rows if o.category == c),
                    sum(1 for o in rows if o.category == c),
                )
                for c in CATEGORIES
            },
            "guards": dict(
                sorted(Counter(o.guard.value if o.guard else "none" for o in rows).items())
            ),
        }
    return summary


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def render_markdown(
    summary: dict[str, Any], outcomes: list[Outcome], tasks_path: str, model: str = "ScriptedLLM"
) -> str:
    variants = list(summary)
    n_tasks = summary[variants[0]]["tasks"]
    counts = Counter(o.category for o in outcomes if o.variant == variants[0])
    lines = [
        "# Evaluation results",
        "",
        f"Generated by `make eval` from `{tasks_path}` ({n_tasks} tasks: "
        + ", ".join(f"{counts[c]} {c}" for c in CATEGORIES)
        + (
            "). Model: `ScriptedLLM` (deterministic replay, no network). "
            "Re-running produces identical numbers."
            if model == "ScriptedLLM"
            else f"). Model: `{model}` (live provider; scripts are ignored, results vary by run; "
            "loop-inducing tasks only induce loops in the scripted model)."
        ),
        "",
        "## Ablation",
        "",
        "| variant | termination rate | task success | correct abstention | mean iterations "
        "| mean tokens | safety-cap hits (loop tasks) |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in variants:
        s = summary[v]
        lines.append(
            f"| {v} | {_pct(s['termination_rate'])} | {_pct(s['success_rate'])} "
            f"| {_pct(s['correct_abstention_rate'])} | {s['mean_iterations']} "
            f"| {s['mean_tokens']} | {s['safety_cap_hits_on_loop_tasks']}/{s['loop_tasks']} |"
        )
    lines += [
        "",
        "## Success by category",
        "",
        "| variant | " + " | ".join(CATEGORIES) + " |",
        "|---|" + "---|" * len(CATEGORIES),
    ]
    for v in variants:
        by_cat = summary[v]["success_by_category"]
        lines.append(f"| {v} | " + " | ".join(_pct(by_cat[c]) for c in CATEGORIES) + " |")
    guards = sorted({g for v in variants for g in summary[v]["guards"]})
    lines += [
        "",
        "## Which guard stopped the run",
        "",
        "`none` = an answer was accepted.",
        "",
        "| variant | " + " | ".join(guards) + " |",
        "|---|" + "---|" * len(guards),
    ]
    for v in variants:
        g = summary[v]["guards"]
        lines.append(f"| {v} | " + " | ".join(str(g.get(k, 0)) for k in guards) + " |")
    lines += [
        "",
        "## Per-task outcomes",
        "",
        "Each cell: status / guard; ✅ = met the task's expectation.",
        "",
        "| task | category | " + " | ".join(variants) + " |",
        "|---|---|" + "---|" * len(variants),
    ]
    by_key = {(o.task_id, o.variant): o for o in outcomes}
    for task_id in dict.fromkeys(o.task_id for o in outcomes):
        cells = []
        for v in variants:
            o = by_key[(task_id, v)]
            mark = "✅" if o.correct else "❌"
            cells.append(f"{mark} {o.status.value} / {o.guard.value if o.guard else 'none'}")
        category = by_key[(task_id, variants[0])].category
        lines.append(f"| {task_id} | {category} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Metric definitions",
        "",
        "- **termination rate**: runs that stopped through the agent's own logic (answer, "
        "abstention, or one of its guards) rather than the baseline's 50-step safety cap or "
        "an internal error.",
        "- **task success**: runs that terminated and whose status matches the task's "
        "expectation in `evals/tasks.yaml`; when the run ends in `success`, its answer must also "
        "match the expected answer.",
        "- **correct abstention**: share of `unsolvable` tasks that ended with status `abstained`.",
        "- **mean tokens**: deterministic estimates (~4 characters per token) of prompt + "
        "completion for every LLM call.",
        "",
    ]
    return "\n".join(lines)


def write_results(
    out_dir: Path,
    summary: dict[str, Any],
    outcomes: list[Outcome],
    tasks_path: str,
    model: str = "ScriptedLLM",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "outcomes": [o.model_dump(mode="json") for o in outcomes],
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out_dir / "results.md").write_text(
        render_markdown(summary, outcomes, tasks_path, model), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the react-agent evaluation.")
    parser.add_argument("--tasks", type=Path, default=Path("evals/tasks.yaml"))
    parser.add_argument("--out", type=Path, default=Path("evals"))
    parser.add_argument("--trace-dir", type=Path, default=Path("traces/eval"))
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument(
        "--provider",
        default="scripted",
        choices=["scripted", "auto", "anthropic", "openai", "ollama"],
        help="scripted (default, free) or a live provider (costs money; needs its env var key)",
    )
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)

    tasks = load_tasks(args.tasks)
    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        parser.error(f"unknown variants: {unknown}; choose from {list(VARIANTS)}")
    variants = {v: VARIANTS[v] for v in args.variants}
    llm_factory: LLMFactory = scripted_llm
    model = "ScriptedLLM"
    if args.provider != "scripted":
        if args.provider == "auto" and available_provider() is None:
            print("eval-real skipped: set ANTHROPIC_API_KEY or OPENAI_API_KEY to run it.")
            return 0
        try:
            probe = make_llm(args.provider, args.model)
        except LLMError as exc:
            print(f"eval-real skipped: {exc}")
            return 0
        model = f"{probe.name}:{args.model or 'default'}"

        def llm_factory(task: TaskSpec) -> LLM:
            return make_llm(args.provider, args.model)

    outcomes = run_eval(tasks, variants, llm_factory=llm_factory, trace_dir=args.trace_dir)
    summary = summarize(outcomes)
    write_results(args.out, summary, outcomes, str(args.tasks), model)
    print((args.out / "results.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
