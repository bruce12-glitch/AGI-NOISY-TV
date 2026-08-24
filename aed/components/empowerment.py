"""Tabular conditional mutual-information / empowerment estimator."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import log
from typing import Any, DefaultDict, Hashable, Mapping

from aed.math_utils import clip
from aed.utils import freeze


class EmpowermentEstimator:
    """Estimate ``I(A; O_next | S)`` from smoothed tabular counts.

    This is intentionally an interpretable one-step estimator. It is not a
    substitute for long-horizon empowerment, but it is sufficient to make the
    TV-versus-puzzle distinction explicit in the proof-of-concept.
    """

    def __init__(self, smoothing: float = 1.0, normalized: bool = True, max_actions: int = 2):
        if smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        self.smoothing = float(smoothing)
        self.normalized = bool(normalized)
        self.max_actions = max(1, int(max_actions))
        self._joint: DefaultDict[Hashable, Counter] = defaultdict(Counter)
        self._actions: DefaultDict[Hashable, Counter] = defaultdict(Counter)
        self._observations: DefaultDict[Hashable, Counter] = defaultdict(Counter)

    def reset(self) -> None:
        self._joint.clear()
        self._actions.clear()
        self._observations.clear()

    def observe(self, state: Any, action: int, next_observation: Any) -> None:
        state_key = freeze(state)
        observation_key = freeze(next_observation)
        self._joint[state_key][(int(action), observation_key)] += 1
        self._actions[state_key][int(action)] += 1
        self._observations[state_key][observation_key] += 1

    def estimate(self, state: Any) -> float:
        """Return non-negative normalized mutual information for a state."""
        state_key = freeze(state)
        joint = self._joint.get(state_key)
        if not joint:
            return 0.0
        total = float(sum(joint.values()))
        action_counts = self._actions[state_key]
        observation_counts = self._observations[state_key]
        mi = 0.0
        for (action, observation), count in joint.items():
            p_joint = count / total
            p_action = action_counts[action] / total
            p_observation = observation_counts[observation] / total
            if p_joint and p_action and p_observation:
                mi += p_joint * log(p_joint / (p_action * p_observation))
        # Miller-Madow-style finite-sample correction keeps spurious mutual
        # information from a small independent table from becoming an AED
        # utility signal.
        n_actions = len(action_counts)
        n_observations = len(observation_counts)
        if n_actions > 1 and n_observations > 1:
            mi = max(0.0, mi - ((n_actions - 1) * (n_observations - 1)) / (2.0 * total))
        if not self.normalized:
            return max(0.0, float(mi))
        # H(A) is a conservative normalization and is at most log(|A|).
        action_entropy = 0.0
        for count in action_counts.values():
            p = count / total
            action_entropy -= p * log(p) if p > 0 else 0.0
        if action_entropy <= 1e-12:
            return 0.0
        return clip(mi / action_entropy)

    def state_statistics(self, state: Any) -> Mapping[str, float]:
        state_key = freeze(state)
        total = sum(self._joint.get(state_key, {}).values())
        return {
            "samples": float(total),
            "empowerment": self.estimate(state),
            "unique_actions": float(len(self._actions.get(state_key, {}))),
            "unique_observations": float(len(self._observations.get(state_key, {}))),
        }
