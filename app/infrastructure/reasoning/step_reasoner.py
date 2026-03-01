import re

from app.domain.models import EvidencePassage, ReasoningStep


class StepReasoner:
    def reason(self, question: str, evidence: list[EvidencePassage]) -> list[ReasoningStep]:
        if not evidence:
            return [
                ReasoningStep(
                    step_number=1,
                    description="No evidence was provided, so retrieval returned no passages.",
                    evidence_ids=[],
                )
            ]

        evidence_ids = [passage.id for passage in evidence]
        steps = [
            ReasoningStep(
                step_number=1,
                description="Retrieve top passages that overlap with key terms in the question.",
                evidence_ids=evidence_ids,
            ),
            ReasoningStep(
                step_number=2,
                description="Identify bridge entities and relations from the retrieved passages.",
                evidence_ids=evidence_ids[:2],
            ),
            ReasoningStep(
                step_number=3,
                description="Synthesize a final answer from aligned evidence.",
                evidence_ids=evidence_ids[:1],
            ),
        ]

        if "same city as" in question.lower():
            steps.insert(
                2,
                ReasoningStep(
                    step_number=3,
                    description="Verify the shared city before selecting the final entity.",
                    evidence_ids=evidence_ids,
                ),
            )
            steps[-1] = ReasoningStep(
                step_number=4,
                description="Synthesize a final answer from aligned evidence.",
                evidence_ids=evidence_ids[:1],
            )

        return steps

    def synthesize(
        self,
        question: str,
        evidence: list[EvidencePassage],
        reasoning_steps: list[ReasoningStep],
    ) -> str:
        if not evidence:
            return "I do not have enough context to answer this question."

        if "university" in question.lower():
            university_patterns = [
                r"\bUniversity of [A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*\b",
                r"\b[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)* University\b",
            ]
            for passage in evidence:
                for pattern in university_patterns:
                    match = re.search(pattern, passage.text)
                    if match:
                        return match.group(0)

        sentence_breaker = re.split(r"(?<=[.!?])\s+", evidence[0].text.strip())
        if sentence_breaker and sentence_breaker[0]:
            return sentence_breaker[0]

        return evidence[0].text[:180]
