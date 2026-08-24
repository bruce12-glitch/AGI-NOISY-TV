"""Streaming, dependency-light telemetry writer."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

from aed.utils import ensure_parent


TELEMETRY_FIELDS = [
    "method",
    "seed",
    "step",
    "room",
    "room_name",
    "action",
    "external_reward",
    "prediction_error",
    "learning_progress",
    "empowerment",
    "utility",
    "tau",
    "boredom",
    "mode_shift",
    "epsilon",
    "action_entropy",
    "shaped_reward",
]


class TelemetryWriter:
    """Write stable CSV columns so analysis remains independent of pandas."""

    def __init__(self, path: str | Path, fields: Optional[Iterable[str]] = None):
        self.path = ensure_parent(path)
        self.fields = list(fields or TELEMETRY_FIELDS)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fields, lineterminator="\n")
        self._writer.writeheader()

    def write(self, row: Mapping[str, object]) -> None:
        self._writer.writerow({field: row.get(field, "") for field in self.fields})

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self) -> "TelemetryWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
