"""Plot mean escape time with normal-approximation confidence bars."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def generate(input_path="results/summary/summary.csv", output="figures/fig_escape_time.png"):
    data = pd.read_csv(input_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = data["escape_time_mean"]
    lower = y - data["escape_time_ci95_low"]
    upper = data["escape_time_ci95_high"] - y
    ax.errorbar(data["method"], y, yerr=[lower, upper], fmt="o", capsize=4, color="#7b2cbf")
    ax.set_ylabel("Steps to first leave Room A")
    ax.set_title("Noisy-TV escape time")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return Path(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/summary/summary.csv")
    parser.add_argument("--output", default="figures/fig_escape_time.png")
    args = parser.parse_args(argv)
    print(f"wrote {generate(args.input, args.output)}")


if __name__ == "__main__":
    main()
