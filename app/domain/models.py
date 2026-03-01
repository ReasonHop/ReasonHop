from dataclasses import dataclass


@dataclass(frozen=True)
class EvidencePassage:
    id: str
    text: str
    score: float


@dataclass(frozen=True)
class ReasoningStep:
    step_number: int
    description: str
    evidence_ids: list[str]


@dataclass(frozen=True)
class AnswerResult:
    question: str
    answer: str
    reasoning_steps: list[ReasoningStep]
    evidence: list[EvidencePassage]
