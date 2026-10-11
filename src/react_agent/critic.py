"""Self-critique: decide whether a proposed final answer is supported by the evidence.

- EvidenceCritic (default, deterministic): every number in the answer must appear (up to the
  answer's precision) in the task or a successful tool observation, and at least half of the
  answer's new content words must appear in the evidence.
- LLMCritic: asks a model for an accept/reject verdict.
- CompositeCritic: runs critics in order and rejects on the first rejection.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from react_agent.llm.base import LLM, LLMError
from react_agent.prompts import build_critic_messages
from react_agent.types import Critique

_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_WORD = re.compile(r"[a-z][a-z']*")
_FILLER_TEXT = (
    "the a an and or of to in on at by for from with is are was were be been it its this that "
    "about approximately roughly around nearly almost some times than per answer result "
    "which who what when where there their they has have had high tall long deep"
)
_FILLER = frozenset(_FILLER_TEXT.split())


def _numbers(text: str) -> list[str]:
    return [m.replace(",", "") for m in _NUMBER.findall(text)]


def _stem(word: str) -> str:
    word = word.removesuffix("'s").strip("'")
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _content_words(text: str) -> set[str]:
    words = (_stem(w) for w in _WORD.findall(text.lower()))
    return {w for w in words if len(w) >= 3 and w not in _FILLER}


def _number_grounded(number: str, sources: list[float]) -> bool:
    decimals = len(number.split(".")[1]) if "." in number else 0
    value = float(number)
    tolerance = 0.5 * 10**-decimals + 1e-9
    return any(abs(src - value) <= tolerance for src in sources)


class Critic(Protocol):
    def review(self, task: str, answer: str, evidence: list[str]) -> Critique: ...


class EvidenceCritic:
    """Deterministic grounding check against the task text and successful observations."""

    def __init__(self, min_word_support: float = 0.5) -> None:
        self.min_word_support = min_word_support

    def review(self, task: str, answer: str, evidence: list[str]) -> Critique:
        if not answer.strip():
            return Critique(accepted=False, reason="the answer is empty")
        if not evidence:
            return Critique(
                accepted=False,
                reason="no successful tool observation supports the answer; gather evidence first",
            )
        corpus = "\n".join(evidence)
        known_numbers = [float(n) for n in _numbers(f"{task}\n{corpus}")]
        ungrounded = [n for n in _numbers(answer) if not _number_grounded(n, known_numbers)]
        if ungrounded:
            return Critique(
                accepted=False,
                reason=f"number(s) {', '.join(ungrounded)} do not appear in any observation",
            )
        novel = _content_words(answer) - _content_words(task)
        if novel:
            supported = novel & _content_words(corpus)
            if len(supported) / len(novel) < self.min_word_support:
                missing = ", ".join(sorted(novel - supported))
                return Critique(
                    accepted=False, reason=f"claims not found in the evidence: {missing}"
                )
        return Critique(accepted=True, reason="every claim is grounded in the evidence")


class LLMCritic:
    """Asks a model to accept or reject. Unparseable or failed critiques count as rejections."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def review(self, task: str, answer: str, evidence: list[str]) -> Critique:
        try:
            response = self.llm.complete(
                build_critic_messages(task, answer, evidence), purpose="critic"
            )
        except LLMError as exc:
            return Critique(accepted=False, reason=f"critic unavailable: {exc}")
        tokens = response.total_tokens
        match = re.search(r"\{.*\}", response.text, re.DOTALL)
        try:
            data = json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            data = None
        if not isinstance(data, dict) or data.get("verdict") not in ("accept", "reject"):
            return Critique(accepted=False, reason="critic output unparseable", tokens_used=tokens)
        return Critique(
            accepted=data["verdict"] == "accept",
            reason=str(data.get("reason", "")) or "no reason given",
            tokens_used=tokens,
        )


class CompositeCritic:
    """Runs each critic in order; the first rejection wins. Token usage is summed."""

    def __init__(self, critics: list[Critic]) -> None:
        self.critics = critics

    def review(self, task: str, answer: str, evidence: list[str]) -> Critique:
        tokens = 0
        reasons: list[str] = []
        for critic in self.critics:
            verdict = critic.review(task, answer, evidence)
            tokens += verdict.tokens_used
            if not verdict.accepted:
                return verdict.model_copy(update={"tokens_used": tokens})
            reasons.append(verdict.reason)
        return Critique(accepted=True, reason="; ".join(reasons), tokens_used=tokens)
