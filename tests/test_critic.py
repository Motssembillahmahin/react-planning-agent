import pytest

from react_agent.critic import CompositeCritic, EvidenceCritic, LLMCritic
from react_agent.llm import ScriptedLLM
from react_agent.types import Critique

critic = EvidenceCritic()


@pytest.mark.parametrize(
    ("task", "answer", "evidence"),
    [
        ("What is 17 * 23 + 4?", "395", ["395"]),
        ("How tall is the Eiffel Tower?", "The Eiffel Tower is 330 metres tall.",
         ["[eiffel] Eiffel Tower: ... It is 330 metres tall including its antennas."]),
        ("How many times taller is Everest?", "about 10.69 times", ["10.68719807"]),
        ("Ratio?", "about 11", ["10.68719807"]),
        ("Convert 100 C to F", "100 °C is 212 °F", ["100 C = 212 F"]),
        ("What is the boiling point of water in Fahrenheit?", "Water boils at 212 °F.",
         ["100 celsius = 212 fahrenheit"]),
        ("Where is the office?", "Lisbon", ["office_city = Lisbon"]),
        ("How far is the Moon?", "384,400 km", ["average distance is 384400 kilometres"]),
    ],
)  # fmt: skip
def test_evidence_critic_accepts_grounded_answers(
    task: str, answer: str, evidence: list[str]
) -> None:
    verdict = critic.review(task, answer, evidence)
    assert verdict.accepted, verdict.reason


@pytest.mark.parametrize(
    ("task", "answer", "evidence", "reason"),
    [
        ("t", "   ", ["x"], "empty"),
        ("Boiling point?", "200 °F", [], "no successful tool observation"),
        ("Population of Atlantis in 1200 BC?", "About 20000 people.", ["Atlantis is fictional."],
         "20000"),
        ("Ratio?", "10.6", ["10.68719807", "8849"], "10.6"),
        ("Who painted the Eiffel Tower?", "Gustave Eiffel", ["Eiffel Tower completed 1889"],
         "gustave"),
    ],
)  # fmt: skip
def test_evidence_critic_rejects_unsupported_answers(
    task: str, answer: str, evidence: list[str], reason: str
) -> None:
    verdict = critic.review(task, answer, evidence)
    assert not verdict.accepted
    assert reason in verdict.reason


def test_llm_critic_parses_verdicts_and_counts_tokens() -> None:
    llm = ScriptedLLM(
        [
            'Here: {"verdict": "accept", "reason": "fine"}',
            {"verdict": "reject", "reason": "unsupported"},
            {"verdict": "accept"},
            "no json",
            "{broken json}",
            {"verdict": "maybe"},
        ]
    )
    c = LLMCritic(llm)
    first = c.review("t", "a", ["e"])
    assert first.accepted
    assert first.reason == "fine"
    assert first.tokens_used > 0
    assert not c.review("t", "a", []).accepted
    assert c.review("t", "a", []).reason == "no reason given"
    for _ in range(3):
        assert c.review("t", "a", []).reason == "critic output unparseable"
    assert llm.purposes == ["critic"] * 6
    unavailable = LLMCritic(ScriptedLLM([])).review("t", "a", [])
    assert not unavailable.accepted
    assert "critic unavailable" in unavailable.reason


class _Fixed:
    def __init__(self, accepted: bool, tokens: int) -> None:
        self.verdict = Critique(accepted=accepted, reason=f"fixed {accepted}", tokens_used=tokens)
        self.calls = 0

    def review(self, task: str, answer: str, evidence: list[str]) -> Critique:
        self.calls += 1
        return self.verdict


def test_composite_critic() -> None:
    a, b = _Fixed(True, 3), _Fixed(True, 4)
    ok = CompositeCritic([a, b]).review("t", "x", [])
    assert ok.accepted
    assert ok.tokens_used == 7
    assert ok.reason == "fixed True; fixed True"
    rej, never = _Fixed(False, 5), _Fixed(True, 1)
    bad = CompositeCritic([_Fixed(True, 2), rej, never]).review("t", "x", [])
    assert not bad.accepted
    assert bad.tokens_used == 7
    assert never.calls == 0
