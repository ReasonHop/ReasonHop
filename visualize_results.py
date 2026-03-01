import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize baseline evaluation results")
    parser.add_argument("--input", type=str, default="outputs/baseline_eval.json")
    parser.add_argument("--output-dir", type=str, default="outputs/charts")
    return parser


def _load_results(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    baselines = payload.get("baselines", [])
    if not isinstance(baselines, list) or not baselines:
        raise ValueError("No baseline results found in input JSON.")
    return baselines


def _save_metric_chart(
    baselines: list[dict[str, object]],
    metric_key: str,
    metric_label: str,
    output_file: Path,
    as_percent: bool,
) -> None:
    names = [str(item["name"]) for item in baselines]
    values = [float(item.get(metric_key, 0.0)) for item in baselines]
    if as_percent:
        values = [value * 100 for value in values]

    plt.figure(figsize=(10, 5.5))
    bars = plt.bar(names, values)
    plt.title(metric_label)
    plt.xlabel("Baseline")
    plt.ylabel(metric_label)
    plt.xticks(rotation=10)

    for bar, value in zip(bars, values):
        label = f"{value:.2f}%" if as_percent else f"{value:.2f}"
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()


def _save_grouped_quality_chart(baselines: list[dict[str, object]], output_file: Path) -> None:
    names = [str(item["name"]) for item in baselines]
    em = [float(item.get("exact_match", 0.0)) * 100 for item in baselines]
    f1 = [float(item.get("f1", 0.0)) * 100 for item in baselines]
    title_recall = [float(item.get("supporting_title_recall", 0.0)) * 100 for item in baselines]
    fact_recall = [float(item.get("supporting_fact_recall", 0.0)) * 100 for item in baselines]

    x = list(range(len(names)))
    width = 0.2

    plt.figure(figsize=(11, 6))
    plt.bar([pos - 1.5 * width for pos in x], em, width=width, label="EM")
    plt.bar([pos - 0.5 * width for pos in x], f1, width=width, label="F1")
    plt.bar([pos + 0.5 * width for pos in x], title_recall, width=width, label="Title Recall")
    plt.bar([pos + 1.5 * width for pos in x], fact_recall, width=width, label="Fact Recall")

    plt.title("Quality Metrics by Baseline")
    plt.xlabel("Baseline")
    plt.ylabel("Score (%)")
    plt.xticks(x, names, rotation=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baselines = _load_results(input_path)

    _save_metric_chart(
        baselines,
        metric_key="exact_match",
        metric_label="Exact Match (%)",
        output_file=output_dir / "exact_match.png",
        as_percent=True,
    )
    _save_metric_chart(
        baselines,
        metric_key="f1",
        metric_label="F1 (%)",
        output_file=output_dir / "f1.png",
        as_percent=True,
    )
    _save_metric_chart(
        baselines,
        metric_key="supporting_title_recall",
        metric_label="Supporting Title Recall (%)",
        output_file=output_dir / "supporting_title_recall.png",
        as_percent=True,
    )
    _save_metric_chart(
        baselines,
        metric_key="supporting_fact_recall",
        metric_label="Supporting Fact Recall (%)",
        output_file=output_dir / "supporting_fact_recall.png",
        as_percent=True,
    )
    _save_metric_chart(
        baselines,
        metric_key="avg_latency_ms",
        metric_label="Average Latency (ms)",
        output_file=output_dir / "latency_ms.png",
        as_percent=False,
    )
    _save_grouped_quality_chart(baselines, output_file=output_dir / "quality_overview.png")

    print(f"Saved charts to: {output_dir}")


if __name__ == "__main__":
    main()
