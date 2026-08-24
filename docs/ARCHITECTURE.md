# Architecture

The package is organized around a single telemetry contract.

```text
StochasticCorridor
        |
        v
  BaseAgent.observe -> TabularWorldModel
        |                    |
        |                    +-- prediction error
        |                    +-- LearningProgress
        |                    +-- EmpowermentEstimator
        v
  Q update + telemetry
        ^
        |
 AEDAgent: HomeostaticThreshold -> BoredomAccumulator -> ModeShiftController
```

## Environment

`StochasticCorridor` has three named regions and a compact `(room, puzzle_progress)` state key. Observations are intentionally separated from the diagnostic room label: TV output is random, puzzle output is action-dependent and becomes stable, and the mastered detour is predictable and transient.

## World model

`TabularWorldModel` records counts for `(state, action, observation)`. Prediction error is the complement of the observed likelihood. `EmpowermentEstimator` computes normalized one-step `I(A; O_next | S)` from the same transition stream. The estimator is useful for this controlled experiment but is not a neural or long-horizon empowerment implementation.

## Agents

All agents use the same tabular Q learner and environment. The extrinsic baseline receives only external reward. Curiosity adds scaled prediction error. AED adds signed learning progress and empowerment, and uses a separate boredom controller to alter epsilon after sustained low utility. This separation makes the mode shift inspectable.

## Telemetry

`aed/logging/telemetry.py` defines stable columns. Raw rows are one post-transition sample per step. Summary analysis groups by method and seed before averaging; time steps are never treated as independent statistical replicates.
