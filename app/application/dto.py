from dataclasses import dataclass


@dataclass(frozen=True)
class AskQuestionCommand:
    question: str
    contexts: list[str]
    top_k: int = 3
