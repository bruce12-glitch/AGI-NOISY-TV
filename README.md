# AED-Boredom-AGI

A small, reproducible research package for testing the **Aversive Epistemic Drive (AED)** hypothesis in a tabular stochastic corridor. The environment contains a noisy television trap, a learnable puzzle, and a mastered room. The package compares extrinsic-only Q-learning, prediction-error curiosity, and AED agents with learning progress, empowerment, a homeostatic boredom accumulator, and exploration mode shifts.

> **Scope.** This is a controlled proof-of-concept, not evidence for AGI or a substitute for a theorem proof. The experiment isolates the mechanism in a deliberately interpretable environment so that the implementation, metrics, and statistical analysis can be inspected and reproduced.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m experiments.run_all --seeds 20
```

The last command runs the corridor suite, writes compact telemetry to `results/raw/`, aggregates results in `results/summary/`, and creates the figures in `figures/`. Use fewer seeds while developing, for example `--seeds 3 --steps 500`.

To run only the central experiment:

```bash
python experiments/run_corridor.py --config configs/corridor/aed.yaml --seeds 20 --steps 3000 --output results/raw/corridor_aed.csv
```

To reproduce analysis from already generated files:

```bash
python -m analysis.summarize_results
python -m analysis.statistical_tests
```

## What is implemented

- A tabular `StochasticCorridor` with explicit noisy, learnable, and mastered regions.
- Prediction-error and learning-progress estimators with exponential moving averages.
- A smoothed tabular mutual-information estimator for empowerment / causal leverage.
- Homeostatic thresholding, leaky boredom accumulation, and finite-duration mode shifts.
- Baseline and ablation agents with common Q-learning infrastructure.
- Seeded telemetry, summary tables, confidence intervals, Welch tests, and publication-style plots.
- Unit tests for environment transitions and the core mechanism components.

The default suite deliberately keeps the state and observation spaces small. This makes failures easy to diagnose before extending the project to MiniGrid or a neural RSSM implementation.

## Checked run

A checked run with 20 independent seeds and a 3,000-step horizon produced the following seed-level means (the full values and 95% intervals are in `results/summary/summary.csv`):

| Method | Room A occupancy | Escape time | External reward |
|---|---:|---:|---:|
| AED | 0.012 | 38.2 | 2,843.1 |
| Curiosity | 0.562 | 1,688.2 | 1,311.4 |
| Extrinsic-only | 0.130 | 391.9 | 2,600.3 |

These numbers are task- and configuration-specific, not a universal claim. In `results/summary/statistical_tests.csv`, the AED-versus-curiosity Welch comparisons are significant for reward and escape time; the AED-versus-extrinsic comparison is also significant for this checked run. Rerun the command after any implementation or dependency change before citing the table.

## Repository map

```text
aed/          core components, models, agents, and environments
configs/      versioned YAML experiment settings
experiments/  commands that generate telemetry
analysis/     summaries, statistics, and figures
tests/        unit and mechanism tests
results/      generated compact CSV evidence
figures/      generated PNG figures
docs/         experimental protocol and theory-to-test map
paper/        table/figure integration placeholders
```

## Interpreting the main result

The intended qualitative comparison is that curiosity assigns value to unpredictable TV observations and therefore spends more time in Room A, while AED's learning-progress/empowerment utility collapses there. Boredom accumulates below the threshold and activates a temporary exploration mode. The result is a test of this mechanism under the stated environment and hyperparameters; it should not be described as a universal guarantee.

The escape-time bound is reported as a diagnostic against the simple no-decay quantity `B_max / tau`. Because the implementation includes leakage and a homeostatic threshold, the exact crossing time is not expected to equal that quantity in every run.

## Citation and license

See `CITATION.cff` for the software citation metadata. The project is released under the MIT License.

## Reproducibility notes

- All random seeds are explicit and stored in every telemetry row.
- Configurations, rather than hidden constants in scripts, control experiments.
- Raw telemetry is ignored by Git by default; compact summaries and figures can be regenerated locally. Small files can be force-added when a paper release requires them.
- No claim in the README should be read as an empirical result until the corresponding command has been run and its output inspected.
