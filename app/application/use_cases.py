from app.application.dto import AskQuestionCommand
from app.domain.models import AnswerResult
from app.domain.ports import Reasoner, Retriever


class AnswerQuestionUseCase:
    def __init__(self, retriever: Retriever, reasoner: Reasoner) -> None:
        self._retriever = retriever
        self._reasoner = reasoner

    def execute(self, command: AskQuestionCommand) -> AnswerResult:
        evidence = self._retriever.retrieve(
            question=command.question,
            contexts=command.contexts,
            top_k=command.top_k,
        )
        reasoning_steps = self._reasoner.reason(command.question, evidence)
        answer = self._reasoner.synthesize(command.question, evidence, reasoning_steps)

        return AnswerResult(
            question=command.question,
            answer=answer,
            reasoning_steps=reasoning_steps,
            evidence=evidence,
        )
