"""Run the reproducibility suite and regenerate tables and figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.plot_ablations import generate as plot_ablations
from analysis.plot_boredom_dynamics import generate as plot_boredom
from analysis.plot_escape_time import generate as plot_escape
from analysis.plot_mode_shift_entropy import generate as plot_entropy
from analysis.plot_occupancy import generate as plot_occupancy
from analysis.plot_regret import generate as plot_regret
from analysis.generate_paper_table import generate as generate_paper_table
from analysis.statistical_tests import generate_tests
from analysis.summarize_results import generate_summary, summarize_files, write_csv
from aed.utils import load_config
from experiments.common import run_config_file

ROOT = Path(__file__).resolve().parents[1]


def run_corridor_suite(seeds: int, steps: int) -> None:
    configs = {
        "aed": ROOT / "configs/corridor/aed.yaml",
        "curiosity": ROOT / "configs/corridor/curiosity.yaml",
        "extrinsic": ROOT / "configs/corridor/extrinsic.yaml",
    }
    for method, config_path in configs.items():
        output = ROOT / f"results/raw/corridor_{method}.csv"
        run_config_file(config_path, output, seeds=seeds, steps=steps)
        print(f"wrote {output}")

    boredom_output = ROOT / "results/raw/boredom_dynamics_seed_0.csv"
    run_config_file(configs["aed"], boredom_output, seeds=1, steps=steps)
    print(f"wrote {boredom_output}")


def run_ablation_suite(seeds: int, steps: int) -> None:
    configs = sorted((ROOT / "configs/ablations").glob("*.yaml"))
    for config_path in configs:
        method = load_config(config_path).get("method", config_path.stem)
        output = ROOT / f"results/raw/ablation_{config_path.stem}.csv"
        run_config_file(config_path, output, seeds=seeds, steps=steps)
        print(f"wrote {output}")
    records = []
    for path in sorted((ROOT / "results/raw").glob("ablation_*.csv")):
        seed_rows, summary_rows = summarize_files([path])
        records.extend(seed_rows)
    write_csv(ROOT / "results/summary/ablation_seed_metrics.csv", records)


def regenerate_analysis() -> None:
    _, summary_path = generate_summary(ROOT / "results/raw", ROOT / "results/summary")
    generate_paper_table(summary_path, ROOT / "paper/tables/results_table.tex")
    generate_tests(ROOT / "results/raw", ROOT / "results/summary/statistical_tests.csv")
    plot_occupancy(ROOT / "results/summary/summary.csv", ROOT / "figures/fig_room_occupancy.png")
    plot_escape(ROOT / "results/summary/summary.csv", ROOT / "figures/fig_escape_time.png")
    plot_boredom(ROOT / "results/raw/boredom_dynamics_seed_0.csv", ROOT / "figures/fig_boredom_dynamics.png")
    plot_entropy(ROOT / "results/raw/boredom_dynamics_seed_0.csv", ROOT / "figures/fig_mode_shift_entropy.png")
    plot_regret(ROOT / "results/raw", ROOT / "figures/fig_regret_curves.png")
    if list((ROOT / "results/raw").glob("ablation_*.csv")):
        plot_ablations(ROOT / "results/raw", ROOT / "figures/fig_ablation_boredom.png")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("all", "corridor", "ablations"), default="all")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args(argv)
    if args.seeds < 1 or args.steps < 1:
        parser.error("--seeds and --steps must be positive")
    if args.experiment in {"all", "corridor"}:
        run_corridor_suite(args.seeds, args.steps)
    if args.experiment in {"all", "ablations"}:
        run_ablation_suite(args.seeds, args.steps)
    if args.experiment == "all":
        regenerate_analysis()
    elif args.experiment == "corridor":
        regenerate_analysis()
    else:
        # Ablation-only execution still creates its dedicated figure/table.
        plot_ablations(ROOT / "results/raw", ROOT / "figures/fig_ablation_boredom.png")
    print("suite complete")


if __name__ == "__main__":
    main()
