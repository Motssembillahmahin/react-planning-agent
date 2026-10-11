"""Command line: `react-agent run "<task>"` and `react-agent replay <trace.jsonl>`."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from react_agent.agent import ReActAgent
from react_agent.config import AgentConfig
from react_agent.llm.base import LLM, LLMError
from react_agent.llm.providers import make_llm
from react_agent.llm.scripted import ScriptedLLM
from react_agent.tools import default_registry
from react_agent.trace import read_trace
from react_agent.types import AgentResult, Status

_STATUS_STYLE = {
    "success": "green",
    "partial": "yellow",
    "abstained": "cyan",
    "failed": "red",
}


def _result_panel(result: dict[str, Any]) -> Panel:
    status = str(result.get("status"))
    lines = [
        f"[bold]status[/]     [{_STATUS_STYLE.get(status, 'white')}]{status}[/]",
        f"[bold]answer[/]     {result.get('answer')}",
        f"[bold]guard[/]      {result.get('guard') or '-'}",
        f"[bold]reason[/]     {result.get('reason')}",
        f"[bold]iterations[/] {result.get('iterations')}"
        f"   [bold]tokens[/] {result.get('tokens_used')}"
        f"   [bold]tool errors[/] {result.get('tool_errors')}"
        f"   [bold]rejections[/] {result.get('critic_rejections')}",
        f"[bold]trace[/]      {result.get('trace_path') or '-'}",
    ]
    return Panel("\n".join(lines), title="result", expand=False)


def _load_llm(args: argparse.Namespace) -> LLM:
    if args.script:
        responses = json.loads(Path(args.script).read_text(encoding="utf-8"))
        return ScriptedLLM(responses)
    return make_llm(args.provider, args.model, args.base_url)


def cmd_run(args: argparse.Namespace, console: Console) -> int:
    try:
        llm = _load_llm(args)
    except (LLMError, OSError, json.JSONDecodeError) as exc:
        console.print(f"[red]cannot start:[/] {exc}")
        return 2
    config = AgentConfig(
        max_iterations=args.max_iterations,
        critic_enabled=not args.no_critic,
    )
    trace_path = Path(args.trace_dir) / f"run-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    agent = ReActAgent(llm, default_registry(), config, trace_path=trace_path)
    result: AgentResult = agent.run(args.task)
    if args.json:
        console.print_json(result.model_dump_json())
    else:
        console.print(_result_panel(result.model_dump(mode="json")))
    return 0 if result.status is Status.SUCCESS else 1


def _describe(event: dict[str, Any]) -> str:
    skip = {"seq", "ts", "run_id", "event", "from", "to", "config"}
    parts = [f"{k}={v}" for k, v in event.items() if k not in skip]
    text = ", ".join(parts)
    return text if len(text) <= 160 else text[:157] + "..."


def cmd_replay(args: argparse.Namespace, console: Console) -> int:
    path = Path(args.trace)
    try:
        events = read_trace(path)
    except (OSError, json.JSONDecodeError) as exc:
        console.print(f"[red]cannot read trace:[/] {exc}")
        return 2
    table = Table(title=f"trace {path.name}", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("event")
    table.add_column("transition")
    table.add_column("details", overflow="fold")
    for e in events:
        transition = f"{e.get('from') or '·'} → {e['to']}" if e["event"] == "transition" else ""
        table.add_row(str(e["seq"]), e["event"], transition, _describe(e))
    console.print(table)
    results = [e for e in events if e["event"] == "result"]
    if results:
        console.print(_result_panel(results[-1]))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="react-agent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the agent on a task")
    run.add_argument("task")
    run.add_argument("--script", help="JSON list of model outputs for ScriptedLLM (free, offline)")
    run.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "anthropic", "openai", "ollama"],
        help="live provider when no --script is given (keys from env vars)",
    )
    run.add_argument("--model", default=None)
    run.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint, e.g. Ollama")
    run.add_argument("--max-iterations", type=int, default=12)
    run.add_argument("--no-critic", action="store_true")
    run.add_argument("--trace-dir", default="traces")
    run.add_argument("--json", action="store_true", help="print the raw result as JSON")

    replay = sub.add_parser("replay", help="pretty-print a JSONL trace")
    replay.add_argument("trace")
    return parser


def main(argv: list[str] | None = None, console: Console | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = console or Console()
    if args.command == "run":
        return cmd_run(args, console)
    return cmd_replay(args, console)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
