from functools import lru_cache

from app.application.use_cases import AnswerQuestionUseCase
from app.infrastructure.reasoning.step_reasoner import StepReasoner
from app.infrastructure.retrieval.keyword_retriever import KeywordRetriever


@lru_cache
def get_answer_question_use_case() -> AnswerQuestionUseCase:
    retriever = KeywordRetriever()
    reasoner = StepReasoner()
    return AnswerQuestionUseCase(retriever=retriever, reasoner=reasoner)
