"""Small helpers for aggregating per-step telemetry."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping


def summarize_trajectory(rows: Iterable[Mapping[str, Any]], steps: int | None = None) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    denominator = float(steps or len(rows))
    occupancy = {}
    for room in (0, 1, 2):
        occupancy[f"occ_{room}"] = sum(int(float(row["room"]) == room) for row in rows) / denominator
    escape = next((int(row["step"]) + 1 for row in rows if int(float(row["room"])) != 0), int(denominator))
    return {
        **occupancy,
        "escape_time": float(escape),
        "first_puzzle_contact": float(escape),
        "mode_shifts": float(sum(float(row.get("mode_shift", 0) or 0) for row in rows)),
        "external_reward": float(sum(float(row.get("external_reward", 0) or 0) for row in rows)),
    }
