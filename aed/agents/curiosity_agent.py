"""Prediction-error curiosity baseline."""

from .base_agent import BaseAgent


class CuriosityAgent(BaseAgent):
    name = "curiosity"

    def __init__(self, config: dict | None = None, seed: int = 0, n_actions: int = 3):
        super().__init__(config=config, seed=seed, n_actions=n_actions)
        curiosity_cfg = (config or {}).get("curiosity", {})
        self.prediction_error_scale = float(curiosity_cfg.get("prediction_error_scale", 2.5))

    def intrinsic_reward(self, metrics: dict[str, float]) -> float:
        return self.prediction_error_scale * metrics["prediction_error"]
