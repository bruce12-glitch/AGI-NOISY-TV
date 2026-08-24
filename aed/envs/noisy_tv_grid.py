"""Compatibility alias for the noisy-TV region of the corridor."""

from .stochastic_corridor import StochasticCorridor


class NoisyTVGrid(StochasticCorridor):
    """Use :class:`StochasticCorridor` while retaining the planned name."""
