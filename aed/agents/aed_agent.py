"""Aversive Epistemic Drive agent."""

from __future__ import annotations

from typing import Optional

from aed.components.boredom_accumulator import BoredomAccumulator
from aed.components.homeostatic_threshold import HomeostaticThreshold
from aed.components.mode_shift import ModeShiftController
from aed.agents.base_agent import BaseAgent


class AEDAgent(BaseAgent):
    """Use learning progress and empowerment, with boredom-triggered exploration.

    Boredom is intentionally a control signal rather than an extra reward term:
    this prevents a large negative shaping reward from making the experiment
    conflate avoidance with exploration. The mode shift changes the action
    distribution after sustained low epistemic utility.
    """

    name = "aed"

    def __init__(self, config: Optional[dict] = None, seed: int = 0, n_actions: int = 3):
        super().__init__(config=config, seed=seed, n_actions=n_actions)
        cfg = config or {}
        aed_cfg = cfg.get("aed", {})
        shift_cfg = cfg.get("mode_shift", {})
        self.alpha_learning_progress = float(aed_cfg.get("alpha_learning_progress", 1.0))
        self.alpha_empowerment = float(aed_cfg.get("alpha_empowerment", 1.0))
        self.beta_boredom = float(aed_cfg.get("beta_boredom", 0.2))
        self.boredom_enabled = bool(aed_cfg.get("boredom_enabled", True))
        self.boredom = BoredomAccumulator(
            decay=float(aed_cfg.get("lambda_decay", 0.99)),
            maximum=float(aed_cfg.get("boredom_max", 10.0)),
        )
        self.threshold = HomeostaticThreshold(
            mu=float(aed_cfg.get("threshold_mu", 1.0)),
            epsilon=float(aed_cfg.get("threshold_epsilon", 0.05)),
            eta=float(aed_cfg.get("threshold_eta", 0.995)),
            initial_tau=float(aed_cfg.get("initial_tau", 0.3)),
            minimum_tau=float(aed_cfg.get("minimum_tau", 0.25)),
            enabled=bool(aed_cfg.get("homeostatic_threshold", True)),
        )
        self.mode_shift = ModeShiftController(
            enabled=bool(shift_cfg.get("enabled", True)) and self.boredom_enabled,
            epsilon_boost=float(shift_cfg.get("epsilon_boost", 0.6)),
            duration=int(shift_cfg.get("duration", 40)),
        )

    def reset(self) -> None:
        super().reset()
        self.boredom.reset()
        self.threshold.reset()
        self.mode_shift.reset()

    def current_epsilon(self, step: int) -> float:
        return self.mode_shift.exploration_epsilon(self.epsilon, step)

    def utility(self, metrics: dict[str, float]) -> float:
        return (
            self.alpha_learning_progress * metrics["learning_progress"]
            + self.alpha_empowerment * metrics["empowerment"]
        )

    def intrinsic_reward(self, metrics: dict[str, float]) -> float:
        # The drive's reward is positive epistemic utility. Boredom controls
        # exploration mode in on_metrics, and is reported separately.
        return metrics["utility"]

    def on_metrics(self, metrics: dict[str, float], step: int) -> bool:
        utility = metrics["utility"]
        tau = self.threshold.step(utility)
        if self.boredom_enabled:
            boredom = self.boredom.step(utility, tau)
            crossed = self.boredom.threshold_crossed
        else:
            boredom = 0.0
            crossed = False
        triggered = self.mode_shift.update(step, crossed)
        metrics["tau"] = tau
        metrics["boredom"] = boredom
        return triggered
