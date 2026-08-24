"""Plot compact ablation comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.common import aggregate_seed_metrics, read_rows


def generate(input_dir="results/raw", output="figures/fig_ablation_boredom.png"):
    records = []
    for path in sorted(Path(input_dir).glob("ablation_*.csv")):
        records.extend(aggregate_seed_metrics(read_rows(path)))
    data = pd.DataFrame(records)
    if data.empty:
        raise FileNotFoundError("No ablation telemetry found")
    means = data.groupby("method", as_index=False)[["occ_0", "escape_time", "external_reward"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = [("occ_0", "Room A occupancy"), ("escape_time", "Escape time"), ("external_reward", "External reward")]
    for ax, (column, title) in zip(axes, labels):
        ax.bar(means["method"], means[column], color="#d1495b")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle("AED ablations")
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return Path(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/raw")
    parser.add_argument("--output", default="figures/fig_ablation_boredom.png")
    args = parser.parse_args(argv)
    print(f"wrote {generate(args.input, args.output)}")


if __name__ == "__main__":
    main()
