"""Aggregate per-step telemetry into paper-friendly CSV tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List

from analysis.common import aggregate_seed_metrics, read_rows
from aed.math_utils import mean_confidence_interval
from aed.utils import ensure_parent


METRIC_COLUMNS = [
    "occ_0",
    "occ_1",
    "occ_2",
    "escape_time",
    "first_puzzle_contact",
    "mode_shifts",
    "external_reward",
]


def summarize_files(paths: Iterable[str | Path]) -> tuple[List[dict], List[dict]]:
    all_seed_metrics = []
    for path in paths:
        all_seed_metrics.extend(aggregate_seed_metrics(read_rows(path)))
    grouped = {}
    for row in all_seed_metrics:
        grouped.setdefault(str(row["method"]), []).append(row)
    summary = []
    for method, rows in sorted(grouped.items()):
        result = {"method": method, "n_seeds": len(rows)}
        for metric in METRIC_COLUMNS:
            values = [float(row[metric]) for row in rows]
            mean, low, high = mean_confidence_interval(values)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else 0.0
            result[f"{metric}_ci95_low"] = low
            result[f"{metric}_ci95_high"] = high
        summary.append(result)
    return all_seed_metrics, summary


def write_csv(path: str | Path, rows: List[dict]) -> None:
    path = ensure_parent(path)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate_summary(raw_dir: str | Path = "results/raw", output_dir: str | Path = "results/summary") -> tuple[Path, Path]:
    raw_dir = Path(raw_dir)
    paths = sorted(raw_dir.glob("corridor_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No corridor telemetry found in {raw_dir}")
    seed_rows, summary_rows = summarize_files(paths)
    seed_path = Path(output_dir) / "seed_metrics.csv"
    summary_path = Path(output_dir) / "summary.csv"
    write_csv(seed_path, seed_rows)
    write_csv(summary_path, summary_rows)
    return seed_path, summary_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/raw")
    parser.add_argument("--output", default="results/summary")
    args = parser.parse_args(argv)
    seed_path, summary_path = generate_summary(args.input, args.output)
    print(f"wrote {seed_path}\nwrote {summary_path}")


if __name__ == "__main__":
    main()
