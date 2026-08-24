"""Dependency-light readers and per-seed metric helpers."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


def read_rows(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(row: Mapping[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in (None, ""):
        return float(default)
    return float(value)


def seed_metrics(rows: Iterable[Mapping[str, str]]) -> Dict[str, float | str | int]:
    rows = list(rows)
    if not rows:
        return {}
    rows = sorted(rows, key=lambda row: int(float(row.get("step", 0))))
    method = rows[0].get("method", "unknown")
    seed = int(float(rows[0].get("seed", 0)))
    steps = len(rows)
    occupancy = {
        f"occ_{room}": sum(int(to_float(row, "room") == room) for row in rows) / steps
        for room in (0, 1, 2)
    }
    outside = [row for row in rows if int(to_float(row, "room")) != 0]
    puzzle = [row for row in rows if int(to_float(row, "room")) == 1]
    escape_time = int(float(outside[0]["step"])) + 1 if outside else steps
    first_puzzle = int(float(puzzle[0]["step"])) + 1 if puzzle else steps
    return {
        "method": method,
        "seed": seed,
        "steps": steps,
        **occupancy,
        "escape_time": float(escape_time),
        "first_puzzle_contact": float(first_puzzle),
        "mode_shifts": sum(to_float(row, "mode_shift") for row in rows),
        "external_reward": sum(to_float(row, "external_reward") for row in rows),
    }


def aggregate_seed_metrics(seed_rows: Iterable[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: defaultdict[tuple[str, int], list] = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["method"]), int(row["seed"]))].append(row)
    return [seed_metrics(group) for group in grouped.values()]
