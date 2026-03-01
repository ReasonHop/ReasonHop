from fastapi import APIRouter, Depends

from app.api.dependencies import get_answer_question_use_case
from app.api.schemas import AskQuestionRequest, AskQuestionResponse
from app.application.dto import AskQuestionCommand
from app.application.use_cases import AnswerQuestionUseCase

router = APIRouter(prefix="/qa", tags=["qa"])


@router.post("/ask", response_model=AskQuestionResponse)
def ask_question(
    payload: AskQuestionRequest,
    use_case: AnswerQuestionUseCase = Depends(get_answer_question_use_case),
) -> AskQuestionResponse:
    command = AskQuestionCommand(
        question=payload.question,
        contexts=payload.contexts,
        top_k=payload.top_k,
    )
    result = use_case.execute(command)

    return AskQuestionResponse(
        question=result.question,
        answer=result.answer,
        reasoning_steps=[
            {
                "step_number": step.step_number,
                "description": step.description,
                "evidence_ids": step.evidence_ids,
            }
            for step in result.reasoning_steps
        ],
        evidence=[
            {
                "id": item.id,
                "text": item.text,
                "score": item.score,
            }
            for item in result.evidence
        ],
    )
