"""Shared setup and trajectory code for experiment commands."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from aed.agents import AEDAgent, CuriosityAgent, ExtrinsicAgent
from aed.envs.stochastic_corridor import StochasticCorridor
from aed.logging.telemetry import TELEMETRY_FIELDS, TelemetryWriter
from aed.utils import ensure_parent, load_config


def make_env(config: dict, seed: int) -> StochasticCorridor:
    env_config = config.get("environment_config", {})
    return StochasticCorridor(seed=seed, **env_config)


def make_agent(config: dict, seed: int):
    method = str(config.get("method", "aed")).lower()
    if method in {"aed", "aed_no_boredom", "aed_no_mode_shift", "aed_fixed_tau", "aed_high_threshold"}:
        return AEDAgent(config=config, seed=seed, n_actions=StochasticCorridor.NUM_ACTIONS)
    if method == "curiosity":
        return CuriosityAgent(config=config, seed=seed, n_actions=StochasticCorridor.NUM_ACTIONS)
    if method in {"extrinsic", "q_learning"}:
        return ExtrinsicAgent(config=config, seed=seed, n_actions=StochasticCorridor.NUM_ACTIONS)
    raise ValueError(f"Unknown tabular method: {method}")


def run_seed(config: dict, seed: int, steps: Optional[int] = None) -> List[Dict[str, Any]]:
    """Run one continuing trajectory and return one row per transition."""
    env = make_env(config, seed)
    agent = make_agent(config, seed)
    configured_steps = int(steps if steps is not None else config.get("steps", 3000))
    observation, info = env.reset(seed=seed)
    agent.reset()
    rows: List[Dict[str, Any]] = []
    method = str(config.get("method", agent.name))
    for step in range(configured_steps):
        state = env.state_key()
        action = agent.select_action(state, info=info if step > 0 else {"room": state[0], "escape_action": 1}, step=step)
        next_observation, external_reward, terminated, truncated, info = env.step(action)
        next_state = env.state_key()
        metrics = agent.observe(
            state=state,
            action=action,
            observation=next_observation,
            next_state=next_state,
            external_reward=external_reward,
            step=step,
            done=terminated or truncated,
        )
        row = {
            "method": method,
            "seed": seed,
            "step": step,
            "room": info["room"],
            "room_name": info["room_name"],
            "action": action,
            "external_reward": external_reward,
            **metrics,
        }
        rows.append(row)
        observation = next_observation
        if terminated or truncated:
            break
    env.close()
    return rows


def write_rows(path: str | Path, rows: Iterable[Dict[str, Any]]) -> Path:
    path = ensure_parent(path)
    with TelemetryWriter(path, fields=TELEMETRY_FIELDS) as writer:
        for row in rows:
            writer.write(row)
    return path


def run_many(config: dict, seeds: int, steps: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed in range(int(seeds)):
        rows.extend(run_seed(config, seed=seed, steps=steps))
    return rows


def run_config_file(config_path: str | Path, output: str | Path, seeds: int, steps: Optional[int] = None) -> Path:
    config = load_config(config_path)
    rows = run_many(config, seeds=seeds, steps=steps)
    return write_rows(output, rows)


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
