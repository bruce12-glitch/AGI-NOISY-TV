"""Leaky accumulator for utility below an adaptive threshold."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BoredomAccumulator:
    """Implement ``B[t+1] = lambda * B[t] + max(0, tau - utility)``."""

    decay: float = 0.99
    maximum: float = 10.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        if self.maximum <= 0:
            raise ValueError("maximum must be positive")
        self.value = 0.0
        self.stagnation_time = 0
        self.crossings = 0
        self.crossed = False

    def reset(self) -> None:
        self.value = 0.0
        self.stagnation_time = 0
        self.crossings = 0
        self.crossed = False

    def step(self, utility: float, tau: float) -> float:
        """Advance the accumulator and return the new value.

        ``crossed`` can be read after the call. It is true only on the rising
        threshold crossing, which avoids counting every step spent above the
        threshold as a new mode shift.
        """
        previous = self.value
        deficit = max(0.0, float(tau) - float(utility))
        self.value = self.decay * self.value + deficit
        if deficit > 0:
            self.stagnation_time += 1
        self.crossed = previous < self.maximum <= self.value
        if self.crossed:
            self.crossings += 1
        return float(self.value)

    @property
    def threshold_crossed(self) -> bool:
        return bool(getattr(self, "crossed", False))

    @property
    def below_threshold(self) -> bool:
        return self.value < self.maximum
