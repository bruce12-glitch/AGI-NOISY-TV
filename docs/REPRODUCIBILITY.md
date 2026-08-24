# Reproducibility

1. Use Python 3.10 or newer and create a clean virtual environment.
2. Install the pinned minimum dependency set in `requirements.txt`.
3. Run `python -m pytest -q`.
4. Run `python -m experiments.run_all --seeds 20` from the repository root.
5. Archive `results/summary/`, `figures/`, the configuration files, and the Git commit identifier.

The default run is CPU-only and has no network dependency. Results are generated rather than checked into the source tree because raw telemetry scales with `seeds * steps`. For a publication release, add the compact summary CSVs and figures using the repository's normal Git workflow or attach a tagged release archive.

## Determinism

NumPy generators are seeded independently for each environment and agent. Determinism assumes the same Python/NumPy implementation and no changes to configuration. The environment is continuing; `steps` is the trajectory horizon.

## Failure modes to check

- A missing `results/raw/corridor_*.csv` means analysis has no input.
- An escape time equal to the horizon may be censoring, not a literal escape at the final step.
- If a new environment exposes a hidden state directly to the model, the TV-vs-puzzle comparison no longer tests the intended observation-learning distinction.
- The optional MiniGrid command is intentionally not part of the default suite.
