"""Generate telemetry used for cumulative external reward / regret plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import run_config_file


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/corridor/aed.yaml")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--output", default="results/raw/regret_aed.csv")
    args = parser.parse_args(argv)
    print(f"wrote {run_config_file(args.config, args.output, args.seeds, args.steps)}")


if __name__ == "__main__":
    main()
