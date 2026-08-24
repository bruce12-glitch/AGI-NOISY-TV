"""A compact tabular world model for the Stochastic Corridor."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, DefaultDict, Hashable

from aed.components.empowerment import EmpowermentEstimator
from aed.components.learning_progress import LearningProgress
from aed.utils import freeze


class TabularWorldModel:
    """Estimate observation error, learning progress, and causal leverage."""

    def __init__(
        self,
        learning_progress: LearningProgress | None = None,
        empowerment: EmpowermentEstimator | None = None,
    ) -> None:
        self.learning_progress = learning_progress or LearningProgress()
        self.empowerment = empowerment or EmpowermentEstimator()
        self.observation_counts: DefaultDict[Hashable, Counter] = defaultdict(Counter)
        self.total_counts: Counter = Counter()

    def reset(self) -> None:
        self.learning_progress.reset()
        self.empowerment.reset()
        self.observation_counts.clear()
        self.total_counts.clear()

    def prediction_error(self, state: Any, action: int, observation: Any) -> float:
        """Return a bounded one-step error before incorporating the sample."""
        key = (freeze(state), int(action))
        counts = self.observation_counts.get(key)
        if not counts:
            return 1.0
        total = sum(counts.values())
        likelihood = counts[freeze(observation)] / total
        return float(1.0 - likelihood)

    def observe(self, state: Any, action: int, observation: Any) -> dict[str, float]:
        """Update the model and return metrics for this transition."""
        state_key = freeze(state)
        action = int(action)
        observation_key = freeze(observation)
        error = self.prediction_error(state_key, action, observation_key)
        progress = self.learning_progress.update(error)
        self.observation_counts[(state_key, action)][observation_key] += 1
        self.total_counts[state_key] += 1
        self.empowerment.observe(state_key, action, observation_key)
        leverage = self.empowerment.estimate(state_key)
        return {
            "prediction_error": float(error),
            "learning_progress": float(progress),
            "empowerment": float(leverage),
        }
