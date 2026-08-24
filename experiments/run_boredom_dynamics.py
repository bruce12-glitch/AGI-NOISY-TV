"""Record the internal AED signals for one seed."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aed.utils import load_config
from experiments.common import run_seed, write_rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/corridor/aed.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--output", default="results/raw/boredom_dynamics_seed_0.csv")
    args = parser.parse_args(argv)
    rows = run_seed(load_config(args.config), seed=args.seed, steps=args.steps)
    print(f"wrote {write_rows(args.output, rows)}")


if __name__ == "__main__":
    main()
