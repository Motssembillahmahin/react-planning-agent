"""Prompt construction for the step model and the LLM critic."""

from __future__ import annotations

from dataclasses import dataclass

from react_agent.llm.base import Message

SYSTEM_PROMPT = """You are a careful ReAct agent. Solve the task step by step using tools.

Reply with ONE JSON object and nothing else, in one of these forms:
  {{"thought": "...", "action": {{"tool": "<name>", "args": {{...}}}}}}
  {{"thought": "...", "final_answer": "<answer grounded in observations>"}}
  {{"thought": "...", "abstain": "<why the task cannot be answered>"}}

Rules:
- Only state facts and numbers that appear in tool observations.
- Do not repeat an action that already produced the same observation.
- If the tools cannot answer the task, abstain instead of guessing.

Available tools:
{tools}
"""

CRITIC_PROMPT = """You are a strict critic. Decide whether the proposed answer is fully
supported by the evidence and actually answers the task.

Task: {task}
Proposed answer: {answer}
Evidence (tool observations):
{evidence}

Reply with ONE JSON object: {{"verdict": "accept" | "reject", "reason": "..."}}"""


@dataclass
class Turn:
    """One model output and the observation it produced."""

    model_text: str
    observation: str


def build_messages(
    task: str, tools_description: str, turns: list[Turn], *, history_window: int = 20
) -> list[Message]:
    """System prompt + task + the last `history_window` turns (bounds prompt growth)."""
    messages = [
        Message(role="system", content=SYSTEM_PROMPT.format(tools=tools_description)),
        Message(role="user", content=f"Task: {task}"),
    ]
    omitted = max(0, len(turns) - history_window)
    if omitted:
        messages.append(Message(role="user", content=f"({omitted} earlier steps omitted)"))
    for turn in turns[omitted:]:
        messages.append(Message(role="assistant", content=turn.model_text))
        messages.append(Message(role="user", content=f"Observation: {turn.observation}"))
    return messages


def build_critic_messages(task: str, answer: str, evidence: list[str]) -> list[Message]:
    rendered = "\n".join(f"- {e}" for e in evidence) or "- (none)"
    return [
        Message(
            role="user",
            content=CRITIC_PROMPT.format(task=task, answer=answer, evidence=rendered),
        )
    ]
