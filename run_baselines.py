import argparse
import json
from pathlib import Path

from experiments.baselines.bm25_reader import BM25ReaderBaseline
from experiments.baselines.implicit_reader import ImplicitReaderBaseline
from experiments.baselines.supporting_fact_naive import SupportingFactNaiveBaseline
from experiments.data.hotpot_loader import load_hotpot_samples
from experiments.evaluation.evaluator import BaselineSummary, evaluate_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ReasonHop baselines on HotpotQA")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--top-k-sentences", type=int, default=5)
    parser.add_argument("--output", type=str, default="outputs/baseline_eval.json")
    return parser


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _print_table(summaries: list[BaselineSummary]) -> None:
    headers = [
        "Baseline",
        "Samples",
        "EM",
        "F1",
        "SupportTitleRecall",
        "SupportFactRecall",
        "AvgLatencyMs",
    ]
    rows = [
        [
            summary.name,
            str(summary.samples),
            _format_percent(summary.exact_match),
            _format_percent(summary.f1),
            _format_percent(summary.supporting_title_recall),
            _format_percent(summary.supporting_fact_recall),
            f"{summary.avg_latency_ms:.2f}",
        ]
        for summary in summaries
    ]

    column_widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            column_widths[index] = max(column_widths[index], len(value))

    def print_row(values: list[str]) -> None:
        line = " | ".join(value.ljust(column_widths[idx]) for idx, value in enumerate(values))
        print(line)

    print_row(headers)
    print("-+-".join("-" * width for width in column_widths))
    for row in rows:
        print_row(row)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    samples = load_hotpot_samples(split=args.split, limit=args.limit)
    baselines = [
        ImplicitReaderBaseline(),
        BM25ReaderBaseline(top_k=args.top_k),
        SupportingFactNaiveBaseline(top_k_sentences=args.top_k_sentences),
    ]

    summaries: list[BaselineSummary] = []
    output_payload: dict[str, object] = {
        "dataset": "hotpot_qa/distractor",
        "split": args.split,
        "limit": args.limit,
        "baselines": [],
    }

    for baseline in baselines:
        summary, examples = evaluate_baseline(baseline=baseline, samples=samples)
        summaries.append(summary)
        output_payload["baselines"].append(
            {
                "name": summary.name,
                "samples": summary.samples,
                "exact_match": summary.exact_match,
                "f1": summary.f1,
                "supporting_title_recall": summary.supporting_title_recall,
                "supporting_fact_recall": summary.supporting_fact_recall,
                "avg_latency_ms": summary.avg_latency_ms,
                "examples": examples,
            }
        )

    _print_table(summaries)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"\nSaved detailed results to: {output_path}")


if __name__ == "__main__":
    main()
