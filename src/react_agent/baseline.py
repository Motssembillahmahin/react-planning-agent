"""NaiveReActAgent: the textbook ReAct loop with no guards, used as the evaluation baseline.

It has no critic, no retries, no fallback, no tool timeout, and no loop detection.
Its only stop condition besides an answer is a hard safety cap of 50 steps.
"""

from __future__ import annotations

import time
from pathlib import Path

from react_agent.llm.base import LLM, LLMError
from react_agent.llm.parsing import ParseError, parse_model_step
from react_agent.prompts import Turn, build_messages
from react_agent.tools.base import RetryPolicy, ToolExecutor, ToolRegistry
from react_agent.trace import Tracer
from react_agent.types import AgentResult, GuardName, Phase, Status

SAFETY_CAP = 50


class NaiveReActAgent:
    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry,
        *,
        safety_cap: int = SAFETY_CAP,
        trace_path: Path | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.safety_cap = safety_cap
        self.trace_path = trace_path
        self._executor = ToolExecutor(
            tools, RetryPolicy(max_retries=0, timeout_s=None), use_fallbacks=False
        )

    def run(self, task: str) -> AgentResult:
        tracer = Tracer(self.trace_path)
        tracer.emit("start", agent="naive", task=task, llm=self.llm.name)
        started = time.monotonic()
        turns: list[Turn] = []
        tokens = 0
        calls = 0
        tool_errors = 0

        def finish(
            status: Status, answer: str | None, guard: GuardName | None, reason: str
        ) -> AgentResult:
            tracer.transition(Phase.DONE if guard is None else Phase.DEGRADED, reason=reason)
            result = AgentResult(
                status=status,
                answer=answer,
                guard=guard,
                reason=reason,
                iterations=calls,
                tokens_used=tokens,
                tool_errors=tool_errors,
                elapsed_s=round(time.monotonic() - started, 4),
                trace_path=str(self.trace_path) if self.trace_path else None,
            )
            tracer.emit("result", **result.model_dump(mode="json"))
            return result

        for _ in range(self.safety_cap):
            tracer.transition(Phase.OBSERVE)
            messages = build_messages(task, self.tools.describe(), turns, history_window=10**9)
            tracer.transition(Phase.THINK)
            try:
                response = self.llm.complete(messages)
            except LLMError as exc:
                return finish(Status.FAILED, None, GuardName.LLM_ERROR, f"LLM error: {exc}")
            calls += 1
            tokens += response.total_tokens
            try:
                step = parse_model_step(response.text)
            except ParseError as exc:
                turns.append(Turn(response.text, f"Invalid response: {exc}"))
                continue
            if step.final_answer is not None:
                return finish(Status.SUCCESS, step.final_answer, None, "model gave a final answer")
            if step.abstain is not None:
                return finish(Status.ABSTAINED, None, GuardName.MODEL_ABSTAINED, step.abstain)
            assert step.action is not None
            tracer.transition(Phase.ACT, action=step.action.model_dump())
            result = self._executor.execute(step.action)
            tool_errors += 0 if result.ok else 1
            tracer.emit("tool", **result.model_dump())
            turns.append(Turn(response.text, result.as_observation()))
        return finish(
            Status.FAILED,
            None,
            GuardName.SAFETY_CAP,
            f"hit the safety cap of {self.safety_cap} steps",
        )
