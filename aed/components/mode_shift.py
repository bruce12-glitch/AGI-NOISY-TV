"""Finite-duration exploration mode shifts."""

from __future__ import annotations

from dataclasses import dataclass

from aed.math_utils import clip


@dataclass
class ModeShiftController:
    """Boost exploration when an accumulator crosses its threshold."""

    enabled: bool = True
    epsilon_boost: float = 0.6
    duration: int = 40

    def __post_init__(self) -> None:
        if self.duration < 0:
            raise ValueError("duration must be non-negative")
        self.active_until = -1
        self.trigger_count = 0
        self.last_trigger_step = -1

    def reset(self) -> None:
        self.active_until = -1
        self.trigger_count = 0
        self.last_trigger_step = -1

    def update(self, step: int, threshold_crossed: bool) -> bool:
        """Register a crossing and return whether a new shift was triggered."""
        if self.enabled and threshold_crossed and not self.is_active(step):
            self.active_until = int(step) + self.duration
            self.last_trigger_step = int(step)
            self.trigger_count += 1
            return True
        return False

    def is_active(self, step: int) -> bool:
        return self.enabled and int(step) < self.active_until

    def exploration_epsilon(self, base_epsilon: float, step: int) -> float:
        if self.is_active(step):
            return clip(float(base_epsilon) + self.epsilon_boost)
        return clip(float(base_epsilon))
