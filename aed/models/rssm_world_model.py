"""Extension point for a future neural AED-Dreamer implementation.

The tabular model is intentionally the only model used by the reproducibility
suite. Keeping this interface explicit prevents an unfinished neural model from
being mistaken for an experimental result.
"""


class RSSMWorldModel:
    """Placeholder interface for a recurrent state-space world model."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "The neural RSSM is planned but not part of the tabular reproducibility suite."
        )
