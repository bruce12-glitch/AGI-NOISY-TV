"""Optional MiniGrid adapter placeholder.

MiniGrid is intentionally not a dependency of the core experiment. This
module fails with an actionable message rather than silently substituting a
 different environment.
"""


class MiniGridAEDWrapper:
    def __init__(self, *args, **kwargs):
        try:
            import gymnasium  # noqa: F401
            import minigrid  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install the optional open-ended dependencies with "
                "`pip install gymnasium minigrid` before using MiniGridAEDWrapper."
            ) from exc
        raise NotImplementedError(
            "The optional MiniGrid experiment is not part of the tabular suite yet."
        )
