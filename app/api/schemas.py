from pydantic import BaseModel, Field


class AskQuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural language question")
    contexts: list[str] = Field(
        default_factory=list,
        description="Retrieved or provided passages used as evidence",
    )
    top_k: int = Field(default=3, ge=1, le=20)


class ReasoningStepResponse(BaseModel):
    step_number: int
    description: str
    evidence_ids: list[str]


class EvidenceResponse(BaseModel):
    id: str
    text: str
    score: float


class AskQuestionResponse(BaseModel):
    question: str
    answer: str
    reasoning_steps: list[ReasoningStepResponse]
    evidence: list[EvidenceResponse]
