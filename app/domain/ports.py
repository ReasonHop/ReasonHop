from typing import Protocol

from app.domain.models import EvidencePassage, ReasoningStep


class Retriever(Protocol):
    def retrieve(self, question: str, contexts: list[str], top_k: int) -> list[EvidencePassage]:
        ...


class Reasoner(Protocol):
    def reason(self, question: str, evidence: list[EvidencePassage]) -> list[ReasoningStep]:
        ...

    def synthesize(
        self,
        question: str,
        evidence: list[EvidencePassage],
        reasoning_steps: list[ReasoningStep],
    ) -> str:
        ...
