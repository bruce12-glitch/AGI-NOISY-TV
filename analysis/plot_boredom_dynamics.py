"""Plot boredom, threshold, utility, and mode-shift markers."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def generate(input_path="results/raw/boredom_dynamics_seed_0.csv", output="figures/fig_boredom_dynamics.png"):
    data = pd.read_csv(input_path)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(data["step"], data["boredom"], label="boredom", color="#d1495b")
    axes[0].plot(data["step"], data["tau"], label="homeostatic threshold", color="#edae49")
    axes[0].axhline(10.0, color="black", linestyle="--", linewidth=0.8, label="B max (default)")
    shifts = data.loc[data["mode_shift"] > 0, "step"].tolist()
    for shift in shifts:
        axes[0].axvline(shift, color="#00798c", alpha=0.35)
    axes[0].set_ylabel("Boredom / threshold")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[0].set_title("AED internal dynamics")
    axes[1].plot(data["step"], data["utility"], label="epistemic utility", color="#30638e")
    axes[1].plot(data["step"], data["action_entropy"], label="policy entropy", color="#6a994e")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Normalized signal")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return Path(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/raw/boredom_dynamics_seed_0.csv")
    parser.add_argument("--output", default="figures/fig_boredom_dynamics.png")
    args = parser.parse_args(argv)
    print(f"wrote {generate(args.input, args.output)}")


if __name__ == "__main__":
    main()
