"""Run the Stochastic Corridor for one configured method."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import run_config_file


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML method configuration")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = run_config_file(args.config, args.output, args.seeds, args.steps)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
