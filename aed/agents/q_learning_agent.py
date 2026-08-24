"""Named compatibility class for a plain tabular Q-learning baseline."""

from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    name = "q_learning"
