"""Small shared types used by the tabular AED implementation."""

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Optional, Tuple

StateKey = Tuple[int, int]
Observation = Hashable


@dataclass
class Transition:
    """A transition emitted by an environment runner."""

    state: StateKey
    action: int
    observation: Any
    next_state: StateKey
    external_reward: float
    done: bool = False
    info: Optional[Dict[str, Any]] = None
