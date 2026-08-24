"""Learning-progress estimation from a prediction-error stream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LearningProgress:
    """Estimate progress as the reduction in an EMA of prediction error.

    Positive values mean that the error signal is falling. Keeping the sign is
    useful: a deteriorating model is not silently treated as progress.
    """

    ema_decay: float = 0.95
    clip_value: Optional[float] = 1.0
    normalize: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        self.ema_error: Optional[float] = None
        self.previous_ema: Optional[float] = None
        self.history: list[float] = []

    def reset(self) -> None:
        self.ema_error = None
        self.previous_ema = None
        self.history.clear()

    def update(self, prediction_error: float) -> float:
        """Consume one error and return signed learning progress."""
        error = max(0.0, float(prediction_error))
        self.previous_ema = self.ema_error
        if self.ema_error is None:
            self.ema_error = error
            progress = 0.0
        else:
            self.ema_error = self.ema_decay * self.ema_error + (1.0 - self.ema_decay) * error
            progress = self.previous_ema - self.ema_error
        if self.normalize:
            denominator = max(abs(self.previous_ema or 0.0), 1e-8)
            progress /= denominator
        if self.clip_value is not None:
            progress = float(np.clip(progress, -self.clip_value, self.clip_value))
        self.history.append(float(progress))
        return float(progress)

    @property
    def current_error(self) -> float:
        return float(self.ema_error or 0.0)
