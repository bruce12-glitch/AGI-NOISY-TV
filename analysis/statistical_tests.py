"""Bootstrap intervals and baseline comparisons across independent seeds."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from analysis.common import read_rows
from analysis.summarize_results import summarize_files
from aed.utils import ensure_parent


def _p_value(x: list[float], y: list[float]) -> float:
    try:
        from scipy.stats import mannwhitneyu, ttest_ind

        # Welch is appropriate for independent seeds with unequal variance.
        return float(ttest_ind(x, y, equal_var=False).pvalue)
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        # Conservative normal approximation fallback.
        mx, my = np.mean(x), np.mean(y)
        vx = np.var(x, ddof=1) if len(x) > 1 else 0.0
        vy = np.var(y, ddof=1) if len(y) > 1 else 0.0
        se = math.sqrt(vx / max(1, len(x)) + vy / max(1, len(y)))
        if se == 0:
            return 0.0 if mx != my else 1.0
        z = abs(mx - my) / se
        return float(math.erfc(z / math.sqrt(2.0)))


def bootstrap_ci(values: Iterable[float], repetitions: int = 2000, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]).tolist())


def generate_tests(raw_dir: str | Path = "results/raw", output: str | Path = "results/summary/statistical_tests.csv") -> Path:
    paths = sorted(Path(raw_dir).glob("corridor_*.csv"))
    seed_rows, _ = summarize_files(paths)
    grouped: Dict[str, List[dict]] = {}
    for row in seed_rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    records = []
    for method in sorted(grouped):
        reward = [float(row["external_reward"]) for row in grouped[method]]
        escape = [float(row["escape_time"]) for row in grouped[method]]
        low, high = bootstrap_ci(reward, seed=17)
        records.append({
            "comparison": method,
            "metric": "external_reward",
            "n": len(reward),
            "mean": float(np.mean(reward)),
            "bootstrap_ci95_low": low,
            "bootstrap_ci95_high": high,
            "reference": "none",
            "p_value": "",
        })
        low, high = bootstrap_ci(escape, seed=23)
        records.append({
            "comparison": method,
            "metric": "escape_time",
            "n": len(escape),
            "mean": float(np.mean(escape)),
            "bootstrap_ci95_low": low,
            "bootstrap_ci95_high": high,
            "reference": "none",
            "p_value": "",
        })
    for metric in ("external_reward", "escape_time"):
        if "aed" not in grouped:
            continue
        for method in sorted(grouped):
            if method == "aed":
                continue
            x = [float(row[metric]) for row in grouped["aed"]]
            y = [float(row[metric]) for row in grouped[method]]
            records.append({
                "comparison": f"aed_vs_{method}",
                "metric": metric,
                "n": min(len(x), len(y)),
                "mean": float(np.mean(x) - np.mean(y)),
                "bootstrap_ci95_low": "",
                "bootstrap_ci95_high": "",
                "reference": method,
                "p_value": _p_value(x, y),
            })
    output = ensure_parent(output)
    with output.open("w", newline="", encoding="utf-8") as handle:
        fields = ["comparison", "metric", "n", "mean", "bootstrap_ci95_low", "bootstrap_ci95_high", "reference", "p_value"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/raw")
    parser.add_argument("--output", default="results/summary/statistical_tests.csv")
    args = parser.parse_args(argv)
    print(f"wrote {generate_tests(args.input, args.output)}")


if __name__ == "__main__":
    main()
