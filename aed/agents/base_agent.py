"""Common tabular Q-learning agent infrastructure."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Optional

import numpy as np

from aed.math_utils import epsilon_greedy_probabilities, normalized_entropy
from aed.models.tabular_world_model import TabularWorldModel
from aed.utils import deep_get


class BaseAgent:
    """A seeded epsilon-greedy tabular agent with shared telemetry hooks."""

    name = "base"

    def __init__(self, config: Optional[dict] = None, seed: int = 0, n_actions: int = 3):
        self.config = config or {}
        self.n_actions = int(n_actions)
        agent_cfg = self.config.get("agent", {})
        self.learning_rate = float(agent_cfg.get("learning_rate", 0.2))
        self.discount_factor = float(agent_cfg.get("discount_factor", 0.95))
        self.epsilon = float(agent_cfg.get("epsilon", 0.05))
        self.rng = np.random.default_rng(seed)
        self.q_values: defaultdict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )
        self.world_model = TabularWorldModel()
        self.last_metrics: Dict[str, float] = self.empty_metrics()
        self.last_mode_shift = False
        self.mode_shift_count = 0

    @staticmethod
    def empty_metrics() -> Dict[str, float]:
        return {
            "prediction_error": 0.0,
            "learning_progress": 0.0,
            "empowerment": 0.0,
            "utility": 0.0,
            "tau": 0.0,
            "boredom": 0.0,
            "mode_shift": 0.0,
            "epsilon": 0.0,
            "action_entropy": 0.0,
            "shaped_reward": 0.0,
        }

    def reset(self) -> None:
        self.q_values.clear()
        self.world_model.reset()
        self.last_metrics = self.empty_metrics()
        self.last_mode_shift = False
        self.mode_shift_count = 0

    def state_key(self, state: Any) -> Any:
        return state

    def current_epsilon(self, step: int) -> float:
        return self.epsilon

    def select_action(self, state: Any, info: Optional[dict] = None, step: int = 0) -> int:
        """Choose an action; deterministic tie-breaking makes seed effects legible."""
        state = self.state_key(state)
        values = self.q_values[state]
        epsilon = self.current_epsilon(step)
        if self.rng.random() < epsilon:
            action = int(self.rng.integers(self.n_actions))
        else:
            # Choosing the first tied action prevents an all-zero table from
            # silently becoming uniformly exploratory before epsilon applies.
            action = int(np.argmax(values))
        probabilities = epsilon_greedy_probabilities(values.tolist(), epsilon)
        self.last_metrics["epsilon"] = epsilon
        self.last_metrics["action_entropy"] = normalized_entropy(probabilities)
        return action

    def intrinsic_reward(self, metrics: Dict[str, float]) -> float:
        return 0.0

    def on_metrics(self, metrics: Dict[str, float], step: int) -> bool:
        """Hook for stateful drives. Return whether a mode shift occurred."""
        return False

    def observe(
        self,
        state: Any,
        action: int,
        observation: Any,
        next_state: Any,
        external_reward: float,
        step: int,
        done: bool = False,
    ) -> Dict[str, float]:
        metrics = self.world_model.observe(state, action, observation)
        metrics["utility"] = self.utility(metrics)
        mode_shift = self.on_metrics(metrics, step)
        shaped_reward = float(external_reward) + self.intrinsic_reward(metrics)
        if done:
            target = shaped_reward
        else:
            target = shaped_reward + self.discount_factor * float(np.max(self.q_values[self.state_key(next_state)]))
        state_key = self.state_key(state)
        old_value = self.q_values[state_key][int(action)]
        self.q_values[state_key][int(action)] = old_value + self.learning_rate * (target - old_value)
        metrics["shaped_reward"] = shaped_reward
        metrics["mode_shift"] = float(mode_shift)
        self.last_mode_shift = bool(mode_shift)
        if mode_shift:
            self.mode_shift_count += 1
        self.last_metrics.update(metrics)
        return dict(self.last_metrics)

    def utility(self, metrics: Dict[str, float]) -> float:
        return 0.0

    def diagnostics(self) -> Dict[str, float]:
        return dict(self.last_metrics)
