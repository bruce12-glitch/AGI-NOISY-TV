"""Plot policy entropy around mode-shift events."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def generate(input_path="results/raw/boredom_dynamics_seed_0.csv", output="figures/fig_mode_shift_entropy.png"):
    data = pd.read_csv(input_path)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(data["step"], data["action_entropy"], color="#6a994e")
    for shift in data.loc[data["mode_shift"] > 0, "step"]:
        ax.axvline(shift, color="#00798c", alpha=0.45, linestyle="--")
    ax.set(xlabel="Step", ylabel="Normalized action entropy", title="Policy entropy and mode shifts")
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return Path(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/raw/boredom_dynamics_seed_0.csv")
    parser.add_argument("--output", default="figures/fig_mode_shift_entropy.png")
    args = parser.parse_args(argv)
    print(f"wrote {generate(args.input, args.output)}")


if __name__ == "__main__":
    main()
