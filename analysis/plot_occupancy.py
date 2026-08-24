"""Plot room occupancy from ``summary.csv``."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate(input_path="results/summary/summary.csv", output="figures/fig_room_occupancy.png"):
    data = pd.read_csv(input_path)
    methods = data["method"].tolist()
    x = np.arange(len(methods))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [("occ_0_mean", "Room A: noisy TV"), ("occ_1_mean", "Room B: puzzle"), ("occ_2_mean", "Room C: mastered")]
    for index, (column, label) in enumerate(labels):
        ax.bar(x + (index - 1) * width, data[column], width, label=label)
    ax.set_xticks(x, methods)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of transitions")
    ax.set_title("Room occupancy by agent")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return Path(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/summary/summary.csv")
    parser.add_argument("--output", default="figures/fig_room_occupancy.png")
    args = parser.parse_args(argv)
    print(f"wrote {generate(args.input, args.output)}")


if __name__ == "__main__":
    main()
