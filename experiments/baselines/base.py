from dataclasses import dataclass
from typing import Protocol

from experiments.data.hotpot_loader import HotpotSample


@dataclass(frozen=True)
class BaselinePrediction:
    answer: str
    retrieved_titles: list[str]
    retrieved_facts: list[tuple[str, int]]


class QABaseline(Protocol):
    name: str

    def predict(self, sample: HotpotSample) -> BaselinePrediction:
        ...
