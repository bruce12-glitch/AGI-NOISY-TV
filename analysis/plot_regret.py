"""Plot mean cumulative external reward by method."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate(input_dir="results/raw", output="figures/fig_regret_curves.png"):
    directory = Path(input_dir)
    paths = sorted(directory.glob("corridor_*.csv"))
    # Support the standalone run_regret.py output without duplicating a method
    # when the full corridor telemetry is already present.
    corridor_methods = set()
    for path in paths:
        data = pd.read_csv(path, nrows=1)
        if not data.empty:
            corridor_methods.add(str(data["method"].iloc[0]))
    for path in sorted(directory.glob("regret_*.csv")):
        data = pd.read_csv(path, nrows=1)
        if not data.empty and str(data["method"].iloc[0]) not in corridor_methods:
            paths.append(path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for path in paths:
        data = pd.read_csv(path)
        if data.empty:
            continue
        data["cumulative_reward"] = data.groupby("seed")["external_reward"].cumsum()
        grouped = data.groupby("step")["cumulative_reward"]
        means = grouped.mean()
        lows = grouped.quantile(0.1)
        highs = grouped.quantile(0.9)
        ax.plot(means.index, means.values, label=str(data["method"].iloc[0]))
        ax.fill_between(means.index, lows.values, highs.values, alpha=0.12)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative external reward")
    ax.set_title("Reward accumulation across seeds")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return Path(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/raw")
    parser.add_argument("--output", default="figures/fig_regret_curves.png")
    args = parser.parse_args(argv)
    print(f"wrote {generate(args.input, args.output)}")


if __name__ == "__main__":
    main()
