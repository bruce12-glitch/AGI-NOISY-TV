"""Numerical helpers with no environment-specific assumptions."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def entropy(probabilities: Sequence[float]) -> float:
    """Shannon entropy in nats, ignoring zero-probability entries."""
    return float(-sum(p * math.log(p) for p in probabilities if p > 0.0))


def normalized_entropy(probabilities: Sequence[float]) -> float:
    """Entropy divided by the maximum possible entropy for this action set."""
    if len(probabilities) <= 1:
        return 0.0
    return entropy(probabilities) / math.log(len(probabilities))


def epsilon_greedy_probabilities(values: Sequence[float], epsilon: float) -> list[float]:
    """Return the exact epsilon-greedy distribution used by the agents."""
    n = len(values)
    if n == 0:
        raise ValueError("At least one action is required")
    greedy = max(range(n), key=lambda index: values[index])
    epsilon = clip(epsilon)
    probabilities = [epsilon / n] * n
    probabilities[greedy] += 1.0 - epsilon
    return probabilities


def mean_confidence_interval(values: Iterable[float], z: float = 1.96) -> tuple[float, float, float]:
    """Normal-approximation mean and 95% interval for compact summaries."""
    numbers = [float(value) for value in values]
    if not numbers:
        return float("nan"), float("nan"), float("nan")
    mean = sum(numbers) / len(numbers)
    if len(numbers) == 1:
        return mean, mean, mean
    variance = sum((value - mean) ** 2 for value in numbers) / (len(numbers) - 1)
    half_width = z * math.sqrt(variance / len(numbers))
    return mean, mean - half_width, mean + half_width
