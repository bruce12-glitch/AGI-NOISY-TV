"""The interpretable stochastic corridor used by the core experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


class Room:
    NOISY_TV = 0
    PUZZLE = 1
    MASTERED = 2


@dataclass
class CorridorState:
    room: int = Room.NOISY_TV
    puzzle_progress: int = 0
    step: int = 0


class StochasticCorridor:
    """A three-region continuing environment.

    * Room A (``NOISY_TV``): actions 0 and 2 watch an unpredictable TV and
      yield no extrinsic reward. Action 1 exits to the puzzle.
    * Room B (``PUZZLE``): actions 0 and 2 work on a learnable puzzle and
      yield a stable external reward; action 1 moves to the mastered room.
    * Room C (``MASTERED``): predictable and unrewarding; navigation actions
      return to the puzzle while the work action can remain for one step.

    The TV randomizes observations without changing the transition. The
    puzzle observation is action-dependent and becomes predictable as its
    progress is learned. ``info`` exposes room labels for telemetry only; the
    agent's state key remains compact and tabular.
    """

    WATCH_TV = 0
    MOVE_CORRIDOR = 1
    WORK_PUZZLE = 2
    NUM_ACTIONS = 3

    def __init__(
        self,
        tv_vocab: int = 8,
        puzzle_mastery_steps: int = 5,
        puzzle_reward: float = 1.0,
        seed: int = 0,
    ) -> None:
        if tv_vocab < 2:
            raise ValueError("tv_vocab must be at least 2")
        self.tv_vocab = int(tv_vocab)
        self.puzzle_mastery_steps = int(max(1, puzzle_mastery_steps))
        self.puzzle_reward = float(puzzle_reward)
        self.rng = np.random.default_rng(seed)
        self.state = CorridorState()
        self._last_room = Room.NOISY_TV

    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = CorridorState()
        self._last_room = self.state.room
        observation = ("tv", int(self.rng.integers(self.tv_vocab)))
        return observation, self._info(action=None, observation=observation)

    def state_key(self) -> Tuple[int, int]:
        return int(self.state.room), int(self.state.puzzle_progress)

    def _info(self, action: Optional[int], observation: Any) -> Dict[str, Any]:
        return {
            "room": int(self.state.room),
            "room_name": self.room_name(self.state.room),
            "puzzle_progress": int(self.state.puzzle_progress),
            "step": int(self.state.step),
            "observation": observation,
            "action": action,
            "escape_action": self.MOVE_CORRIDOR if self.state.room == Room.NOISY_TV else None,
            "entered_room": self.state.room if self.state.room != self._last_room else None,
        }

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        action = int(action)
        if action < 0 or action >= self.NUM_ACTIONS:
            raise ValueError(f"action must be in [0, {self.NUM_ACTIONS})")
        previous_room = self.state.room
        reward = 0.0

        if self.state.room == Room.NOISY_TV:
            if action == self.MOVE_CORRIDOR:
                self.state.room = Room.PUZZLE
            # Every non-exit action observes independent TV noise.
            observation = ("tv", int(self.rng.integers(self.tv_vocab)))
        elif self.state.room == Room.PUZZLE:
            if action == self.MOVE_CORRIDOR:
                self.state.room = Room.MASTERED
                # The room transition is not itself a noisy observation. The
                # model sees a predictable puzzle cue, while the diagnostic
                # room label records the mastered detour.
                observation = ("puzzle", self.state.puzzle_progress)
            elif action in (self.WATCH_TV, self.WORK_PUZZLE):
                # Two work affordances make the learnable room discoverable by
                # a plain epsilon-greedy baseline, without changing the TV
                # trap where both non-exit actions remain unrewarded.
                self.state.puzzle_progress = min(
                    self.puzzle_mastery_steps, self.state.puzzle_progress + 1
                )
                reward = self.puzzle_reward
                # The action tag is a controllable cue, so the tabular
                # empowerment estimator can distinguish puzzle affordances
                # even after the latent puzzle is mastered.
                observation = ("puzzle", int(action), self.state.puzzle_progress)
            else:
                observation = ("puzzle", "exit", self.state.puzzle_progress)
        else:  # mastered room
            # The mastered room is a brief predictable detour. Both
            # navigation actions return to the still-useful puzzle; the work
            # action can remain in the mastered state for a stagnation probe.
            if action != self.WORK_PUZZLE:
                self.state.room = Room.PUZZLE
                observation = ("puzzle", self.state.puzzle_progress)
            else:
                observation = ("maze", 0)

        self.state.step += 1
        self._last_room = previous_room
        info = self._info(action=action, observation=observation)
        info["entered_room"] = self.state.room if self.state.room != previous_room else None
        return observation, float(reward), False, False, info

    @staticmethod
    def room_name(room: int) -> str:
        return {
            Room.NOISY_TV: "noisy_tv",
            Room.PUZZLE: "puzzle",
            Room.MASTERED: "mastered",
        }.get(int(room), "unknown")

    def close(self) -> None:
        """Gymnasium-compatible no-op."""


# Friendly aliases for papers and notebooks.
NOISY_TV = Room.NOISY_TV
LEARNABLE_PUZZLE = Room.PUZZLE
MASTERED_MAZE = Room.MASTERED
