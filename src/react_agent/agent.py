"""ReActAgent: an explicit state machine OBSERVE → THINK → ACT → REFLECT (→ CRITIQUE) → DONE.

Termination argument (checked by the property test):
- Every pass through THINK increments `iterations`, and BEFORE_THINK refuses to start a pass
  once `iterations >= max_iterations`. Every other edge either returns or leads back to
  OBSERVE → THINK, so the loop makes at most `max_iterations` model calls for steps.
- The critic runs at most once per proposed answer and the run ends at the
  `max_critic_rejections`-th rejection, so critic calls are bounded by the same budget plus the cap.
- Tool calls are bounded by the executor's retry count and per-attempt timeout.
- `run()` catches everything else and returns a DEGRADED result, so it never raises.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from react_agent.config import AgentConfig
from react_agent.critic import Critic, EvidenceCritic
from react_agent.guards import Checkpoint, GuardTrip, LoopState, action_fingerprint, build_guards
from react_agent.llm.base import LLM, LLMError
from react_agent.llm.parsing import ParseError, parse_model_step
from react_agent.prompts import Turn, build_messages
from react_agent.text import info_tokens
from react_agent.tools.base import ToolExecutor, ToolRegistry
from react_agent.trace import Tracer
from react_agent.types import Action, AgentResult, GuardName, Phase, Status, ToolResult

# Guards that mean "the agent chose not to (or could not responsibly) answer".
_ABSTAIN_GUARDS = frozenset(
    {
        GuardName.MODEL_ABSTAINED,
        GuardName.CRITIC_REJECTIONS,
        GuardName.REPEATED_ACTION,
        GuardName.NO_PROGRESS,
    }
)


def status_for(guard: GuardName, best_answer: str | None) -> Status:
    """Map the guard that stopped a run (plus any candidate answer) to a result status."""
    if guard in (GuardName.MODEL_ABSTAINED, GuardName.CRITIC_REJECTIONS):
        return Status.ABSTAINED
    if best_answer is not None:
        return Status.PARTIAL
    return Status.ABSTAINED if guard in _ABSTAIN_GUARDS else Status.FAILED


class _Run:
    """Mutable state of a single run."""

    def __init__(self, task: str, started_at: float) -> None:
        self.task = task
        self.state = LoopState(started_at=started_at)
        self.turns: list[Turn] = []
        self.evidence: list[str] = []
        self.seen: set[str] = info_tokens(task)
        self.best_answer: str | None = None


class ReActAgent:
    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
        *,
        critic: Critic | None = None,
        trace_path: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.config = config or AgentConfig()
        self.critic: Critic | None = (
            (critic or EvidenceCritic()) if self.config.critic_enabled else None
        )
        self.trace_path = trace_path
        self._clock = clock
        self.guards = build_guards(self.config, clock)
        self.executor = ToolExecutor(tools, self.config.retry_policy(), sleep=sleep)

    # ------------------------------------------------------------------ public API

    def run(self, task: str) -> AgentResult:
        """Run the loop. Never raises: every failure becomes a structured AgentResult."""
        tracer = Tracer(self.trace_path)
        run = _Run(task, self._clock())
        try:
            tracer.emit(
                "start",
                agent="react",
                task=task,
                llm=self.llm.name,
                config=self.config.model_dump(),
                guards=self.guards.names(),
                critic=type(self.critic).__name__ if self.critic else None,
            )
            return self._loop(run, tracer)
        except Exception as exc:  # last line of defence: degrade, don't raise
            trip = GuardTrip(GuardName.INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
            try:
                return self._degrade(run, tracer, trip, status=Status.FAILED)
            except Exception:  # even the trace is broken: return without touching it
                return AgentResult(
                    status=Status.FAILED,
                    guard=GuardName.INTERNAL_ERROR,
                    reason=trip.reason,
                    iterations=run.state.iterations,
                    tokens_used=run.state.tokens_used,
                )

    # ------------------------------------------------------------------ the state machine

    def _loop(self, run: _Run, tracer: Tracer) -> AgentResult:
        state = run.state
        while True:
            tracer.transition(Phase.OBSERVE, iteration=state.iterations)
            trip = self.guards.check(Checkpoint.BEFORE_THINK, state)
            if trip:
                return self._degrade(run, tracer, trip)
            messages = build_messages(
                run.task,
                self.tools.describe(),
                run.turns,
                history_window=self.config.history_window,
            )

            tracer.transition(Phase.THINK, iteration=state.iterations + 1)
            state.iterations += 1
            try:
                response = self.llm.complete(messages, purpose="step")
            except LLMError as exc:
                return self._degrade(run, tracer, GuardTrip(GuardName.LLM_ERROR, str(exc)))
            state.tokens_used += response.total_tokens
            tracer.emit("llm", text=response.text, tokens=response.total_tokens)

            try:
                step = parse_model_step(response.text)
            except ParseError as exc:
                observation = f"Invalid response ({exc}). Reply with exactly one JSON object."
                run.turns.append(Turn(response.text, observation))
                trip = self._reflect(run, tracer, new_tokens=set(), note=observation)
                if trip:
                    return self._degrade(run, tracer, trip)
                continue

            if step.abstain is not None:
                trip = GuardTrip(GuardName.MODEL_ABSTAINED, step.abstain)
                return self._finish(run, tracer, Status.ABSTAINED, None, trip)

            if step.final_answer is not None:
                result = self._critique(run, tracer, response.text, step.final_answer)
                if result is not None:
                    return result
                continue

            assert step.action is not None  # guaranteed by ModelStep validation
            trip = self.guards.check(Checkpoint.BEFORE_ACT, state, step.action)
            if trip:
                return self._degrade(run, tracer, trip)
            tool_result = self._act(run, tracer, step.action, response.text)
            trip = self.guards.check(Checkpoint.AFTER_ACT, state)
            if trip:
                return self._degrade(run, tracer, trip)
            new = (
                info_tokens(tool_result.output) - info_tokens(str(step.action.args))
                if tool_result.ok
                else set()
            )
            trip = self._reflect(run, tracer, new_tokens=new, note=tool_result.as_observation())
            if trip:
                return self._degrade(run, tracer, trip)

    def _act(self, run: _Run, tracer: Tracer, action: Action, model_text: str) -> ToolResult:
        tracer.transition(Phase.ACT, tool=action.tool, args=action.args)
        run.state.action_history.append(action_fingerprint(action))
        result = self.executor.execute(action)
        tracer.emit("tool", **result.model_dump())
        if result.ok:
            run.evidence.append(result.output)
        else:
            run.state.tool_errors += 1
        run.turns.append(Turn(model_text, result.as_observation()))
        return result

    def _reflect(
        self, run: _Run, tracer: Tracer, *, new_tokens: set[str], note: str
    ) -> GuardTrip | None:
        """Decide whether the last step added information; feed the no-progress detector."""
        fresh = new_tokens - run.seen
        run.seen |= new_tokens
        state = run.state
        state.no_progress_streak = 0 if fresh else state.no_progress_streak + 1
        tracer.transition(
            Phase.REFLECT,
            new_information=bool(fresh),
            new_tokens=sorted(fresh)[:20],
            no_progress_streak=state.no_progress_streak,
            observation=note[:500],
        )
        return self.guards.check(Checkpoint.AFTER_REFLECT, state)

    def _critique(
        self, run: _Run, tracer: Tracer, model_text: str, answer: str
    ) -> AgentResult | None:
        """Return a final result, or None to keep looping with the critique as feedback."""
        tracer.transition(Phase.CRITIQUE, answer=answer)
        run.best_answer = answer
        if self.critic is None:
            return self._finish(run, tracer, Status.SUCCESS, answer, None, "critic disabled")
        verdict = self.critic.review(run.task, answer, run.evidence)
        run.state.tokens_used += verdict.tokens_used
        tracer.emit("critique", answer=answer, **verdict.model_dump())
        if verdict.accepted:
            return self._finish(run, tracer, Status.SUCCESS, answer, None, verdict.reason)
        run.state.critic_rejections += 1
        if run.state.critic_rejections >= self.config.max_critic_rejections:
            reason = (
                f"critic rejected {run.state.critic_rejections} answers; "
                f"last {answer!r}: {verdict.reason}"
            )
            return self._degrade(run, tracer, GuardTrip(GuardName.CRITIC_REJECTIONS, reason))
        run.turns.append(
            Turn(
                model_text,
                f"Critic rejected the answer: {verdict.reason}. "
                "Gather supporting evidence with a tool, or abstain.",
            )
        )
        return None

    # ------------------------------------------------------------------ endings

    def _degrade(
        self, run: _Run, tracer: Tracer, trip: GuardTrip, *, status: Status | None = None
    ) -> AgentResult:
        status = status or status_for(trip.guard, run.best_answer)
        answer = run.best_answer if status is Status.PARTIAL else None
        tracer.transition(Phase.DEGRADED, guard=trip.guard.value, reason=trip.reason)
        return self._result(run, tracer, status, answer, trip.guard, trip.reason)

    def _finish(
        self,
        run: _Run,
        tracer: Tracer,
        status: Status,
        answer: str | None,
        trip: GuardTrip | None,
        reason: str = "",
    ) -> AgentResult:
        reason = trip.reason if trip else reason
        tracer.transition(Phase.DONE, status=status.value, reason=reason)
        return self._result(run, tracer, status, answer, trip.guard if trip else None, reason)

    def _result(
        self,
        run: _Run,
        tracer: Tracer,
        status: Status,
        answer: str | None,
        guard: GuardName | None,
        reason: str,
    ) -> AgentResult:
        state = run.state
        result = AgentResult(
            status=status,
            answer=answer,
            guard=guard,
            reason=reason,
            iterations=state.iterations,
            tokens_used=state.tokens_used,
            tool_errors=state.tool_errors,
            critic_rejections=state.critic_rejections,
            elapsed_s=round(max(0.0, self._clock() - state.started_at), 4),
            trace_path=str(self.trace_path) if self.trace_path else None,
        )
        tracer.emit("result", **result.model_dump(mode="json"))
        return result
