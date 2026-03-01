from dataclasses import dataclass
from time import perf_counter

from experiments.baselines.base import QABaseline
from experiments.data.hotpot_loader import HotpotSample
from experiments.evaluation.metrics import (
    exact_match,
    f1_score,
    supporting_fact_recall,
    supporting_title_recall,
)


@dataclass(frozen=True)
class BaselineSummary:
    name: str
    samples: int
    exact_match: float
    f1: float
    supporting_title_recall: float
    supporting_fact_recall: float
    avg_latency_ms: float


def evaluate_baseline(
    baseline: QABaseline,
    samples: list[HotpotSample],
) -> tuple[BaselineSummary, list[dict[str, str]]]:
    em_total = 0.0
    f1_total = 0.0
    title_recall_total = 0.0
    fact_recall_total = 0.0
    latency_total = 0.0
    examples: list[dict[str, str]] = []

    for sample in samples:
        start = perf_counter()
        prediction = baseline.predict(sample)
        elapsed = perf_counter() - start

        em = exact_match(prediction.answer, sample.answer)
        f1 = f1_score(prediction.answer, sample.answer)
        title_recall = supporting_title_recall(prediction.retrieved_titles, sample.supporting_titles)
        fact_recall = supporting_fact_recall(prediction.retrieved_facts, sample.supporting_facts)

        em_total += em
        f1_total += f1
        title_recall_total += title_recall
        fact_recall_total += fact_recall
        latency_total += elapsed * 1000

        if len(examples) < 5:
            examples.append(
                {
                    "id": sample.sample_id,
                    "question": sample.question,
                    "gold_answer": sample.answer,
                    "prediction": prediction.answer,
                }
            )

    count = max(1, len(samples))
    summary = BaselineSummary(
        name=baseline.name,
        samples=len(samples),
        exact_match=em_total / count,
        f1=f1_total / count,
        supporting_title_recall=title_recall_total / count,
        supporting_fact_recall=fact_recall_total / count,
        avg_latency_ms=latency_total / count,
    )

    return summary, examples
