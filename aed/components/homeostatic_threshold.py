"""Homeostatic threshold for an epistemic utility signal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class HomeostaticThreshold:
    """Track ``tau = mu * EMA(U) + epsilon`` with a positive floor.

    ``initial_tau`` is represented by an initial EMA of
    ``initial_tau - epsilon``. ``minimum_tau`` is a practical guard against a
    long quiet episode lowering the drive threshold to numerical zero; it is
    reported in the configuration and should be included in any formal model.
    """

    mu: float = 1.0
    epsilon: float = 0.05
    eta: float = 0.995
    initial_tau: float = 0.3
    minimum_tau: float = 0.25
    enabled: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.eta < 1.0:
            raise ValueError("eta must be in [0, 1)")
        if self.initial_tau <= 0 or self.minimum_tau <= 0:
            raise ValueError("initial_tau and minimum_tau must be positive")
        if self.minimum_tau > self.initial_tau:
            raise ValueError("minimum_tau cannot exceed initial_tau")
        self.mean_utility: Optional[float] = max(0.0, self.initial_tau - self.epsilon)
        self.value = float(self.initial_tau)

    def reset(self) -> None:
        self.mean_utility = max(0.0, self.initial_tau - self.epsilon)
        self.value = float(self.initial_tau)

    def step(self, utility: float) -> float:
        """Consume utility and return the threshold used for this step."""
        utility = max(0.0, float(utility))
        if self.enabled:
            assert self.mean_utility is not None
            self.mean_utility = self.eta * self.mean_utility + (1.0 - self.eta) * utility
            self.value = max(self.minimum_tau, self.mu * self.mean_utility + self.epsilon)
        else:
            self.value = float(self.initial_tau)
        return float(self.value)
