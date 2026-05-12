import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


DEFAULT_RESULTS_FILE = "benchmark_performance.tsv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize benchmark test runs from a tab-separated results file."
    )
    parser.add_argument(
        "--results-file",
        default=DEFAULT_RESULTS_FILE,
        help=f"Path to the TSV results file (default: {DEFAULT_RESULTS_FILE})",
    )
    return parser.parse_args()


def load_rows(results_path: Path):
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with results_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    if not rows:
        raise ValueError(f"No data rows found in {results_path}")

    required_columns = {"Benchmark", "Configuration", "Recall", "Precision", "Latency_ms"}
    missing = required_columns - set(rows[0].keys())
    if missing:
        raise ValueError(
            f"Missing required columns in {results_path}: {', '.join(sorted(missing))}"
        )

    return rows


def to_float(value, field_name, row_number):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid numeric value for {field_name} on row {row_number}: {value!r}"
        ) from exc


def percentile(values, pct):
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build_groups(rows):
    overall = {
        "recall": [],
        "precision": [],
        "latency": [],
    }
    by_benchmark = defaultdict(lambda: {"recall": [], "precision": [], "latency": []})
    by_configuration = defaultdict(lambda: {"recall": [], "precision": [], "latency": []})

    for idx, row in enumerate(rows, start=2):
        recall = to_float(row.get("Recall"), "Recall", idx)
        precision = to_float(row.get("Precision"), "Precision", idx)
        latency = to_float(row.get("Latency_ms"), "Latency_ms", idx)

        overall["recall"].append(recall)
        overall["precision"].append(precision)
        overall["latency"].append(latency)

        benchmark = row.get("Benchmark", "unknown")
        configuration = row.get("Configuration", "unknown")

        by_benchmark[benchmark]["recall"].append(recall)
        by_benchmark[benchmark]["precision"].append(precision)
        by_benchmark[benchmark]["latency"].append(latency)

        by_configuration[configuration]["recall"].append(recall)
        by_configuration[configuration]["precision"].append(precision)
        by_configuration[configuration]["latency"].append(latency)

    return overall, by_benchmark, by_configuration


def summarize_bucket(values):
    return {
        "count": len(values["recall"]),
        "avg_recall": mean(values["recall"]),
        "avg_precision": mean(values["precision"]),
        "avg_latency_ms": mean(values["latency"]),
        "median_latency_ms": median(values["latency"]),
        "p95_latency_ms": percentile(values["latency"], 0.95),
        "min_latency_ms": min(values["latency"]),
        "max_latency_ms": max(values["latency"]),
    }


def print_summary(title, summary):
    print(title)
    print("-" * len(title))
    print(f"Runs:             {summary['count']}")
    print(f"Avg recall:       {summary['avg_recall']:.4f}")
    print(f"Avg precision:    {summary['avg_precision']:.4f}")
    print(f"Avg latency (ms): {summary['avg_latency_ms']:.2f}")
    print(f"Median latency:   {summary['median_latency_ms']:.2f}")
    print(f"P95 latency:      {summary['p95_latency_ms']:.2f}")
    print(f"Latency range:    {summary['min_latency_ms']:.2f} - {summary['max_latency_ms']:.2f}")
    print()


def print_table(title, grouped):
    print(title)
    print("-" * len(title))
    print(
        f"{'Name':<35} {'Runs':>5} {'Recall':>8} {'Precision':>10} {'Latency ms':>12} {'P95':>10}"
    )
    print("-" * 85)
    for name in sorted(grouped):
        summary = summarize_bucket(grouped[name])
        print(
            f"{name:<35} {summary['count']:>5} "
            f"{summary['avg_recall']:>8.4f} {summary['avg_precision']:>10.4f} "
            f"{summary['avg_latency_ms']:>12.2f} {summary['p95_latency_ms']:>10.2f}"
        )
    print()


def main():
    args = parse_args()
    results_path = Path(args.results_file)

    rows = load_rows(results_path)
    overall, by_benchmark, by_configuration = build_groups(rows)

    print(f"Results file: {results_path.resolve()}")
    print(f"Rows parsed:  {len(rows)}")
    print()

    print_summary("Overall", summarize_bucket(overall))
    print_table("By Benchmark", by_benchmark)
    print_table("By Configuration", by_configuration)


if __name__ == "__main__":
    main()
