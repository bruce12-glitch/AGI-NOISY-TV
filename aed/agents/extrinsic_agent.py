"""Extrinsic-only Q-learning baseline."""

from .q_learning_agent import QLearningAgent


class ExtrinsicAgent(QLearningAgent):
    name = "extrinsic"

    def intrinsic_reward(self, metrics: dict[str, float]) -> float:
        return 0.0
