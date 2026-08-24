"""Explicit opt-in entry point for the future open-ended MiniGrid study."""

from __future__ import annotations

import argparse


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/openended/minigrid_aed.yaml")
    parser.parse_args(argv)
    raise SystemExit(
        "The optional MiniGrid study is not implemented in the tabular release. "
        "Use the Stochastic Corridor suite first; see aed/envs/minigrid_wrapper.py."
    )


if __name__ == "__main__":
    main()
