"""Compatibility alias for the mastered region of the corridor."""

from .stochastic_corridor import StochasticCorridor


class MasteredMaze(StochasticCorridor):
    """Use :class:`StochasticCorridor` while retaining the planned name."""
