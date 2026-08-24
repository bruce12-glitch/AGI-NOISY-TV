# Experimental protocol

## Scope

The release evaluates a mechanism in a deliberately small continuing MDP. Each seed starts in Room A and receives the same action set. The only stochasticity in the default experiment is the TV observation stream and the seeded policy/environment generators.

## E1: stochastic corridor escape

**Hypothesis:** prediction-error curiosity overvalues unlearnable TV noise; AED accumulates boredom when learning progress and one-step empowerment are low, then increases exploration and leaves the trap.

Run:

```bash
python experiments/run_all.py --experiment corridor --seeds 20 --steps 3000
```

Telemetry includes room occupancy, first exit time, first puzzle contact, external reward, utility, threshold, boredom, action entropy, and mode-shift events. Room occupancy is computed over post-transition room labels. If a run never leaves Room A, its escape time is censored at the trajectory length; the summary does not pretend that the censoring is an observed escape.

## E2: reward / regret conversion

The environment's optimal continuing behavior is to work in Room B, where `WORK_PUZZLE` produces the configured external reward. `fig_regret_curves.png` plots cumulative external reward, which is the directly observed quantity. A separate regret estimate should state its assumed optimal reward rate rather than silently treating it as known.

```bash
python experiments/run_regret.py --config configs/corridor/aed.yaml \
  --seeds 20 --steps 5000 --output results/raw/regret_aed.csv
python analysis/plot_regret.py --input results/raw \
  --output figures/fig_regret_curves.png
```

## E3: boredom and mode-shift signature

```bash
python experiments/run_boredom_dynamics.py --seed 0 --steps 3000 \
  --output results/raw/boredom_dynamics_seed_0.csv
python analysis/plot_boredom_dynamics.py
python analysis/plot_mode_shift_entropy.py
```

A vertical marker denotes a threshold crossing. The policy entropy series is the analytic entropy of the epsilon-greedy action distribution used by the agent, not a post-hoc estimate from action counts.

## E4: ablations

```bash
python experiments/run_all.py --experiment ablations --seeds 20 --steps 3000
```

The included ablations remove the boredom control signal, remove mode shifts, hold the threshold fixed, and increase `B_max`. Ablations are descriptive mechanism tests; they are not automatically proof that one component is the only possible causal explanation.

## E5: uncertainty and tests

```bash
python analysis/statistical_tests.py --input results/raw \
  --output results/summary/statistical_tests.csv
```

The analysis reports normal-approximation CIs in `summary.csv` for compact visualization and bootstrap mean CIs plus Welch comparisons in `statistical_tests.csv`. Seeds are independent trajectories, not independent time steps. For a formal paper, preserve the exact analysis version and report any multiple-comparison correction.

## Configuration

Every method and ablation is configured in `configs/`. The default AED threshold is initialized at `0.3`, has an explicit floor of `0.25`, leaks with `lambda_decay=0.99`, and shifts exploration for 40 steps after crossing `B_max=10`. The simple diagnostic `B_max / tau` is a no-decay reference. Leakage and homeostasis are intentionally retained in the actual agent, so equality with that diagnostic is not expected.

## Reproducibility checklist

- Run from the repository root.
- Record the Git commit, Python version, dependency versions, and CLI arguments in a paper artifact.
- Use at least 20 seeds for a reported comparison.
- Inspect raw telemetry for failed/censored runs before interpreting means.
- Regenerate figures from CSV rather than editing labels or numbers by hand.
