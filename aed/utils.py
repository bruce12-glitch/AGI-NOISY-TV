"""Utilities shared by experiments and agents."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - requirements install PyYAML
    yaml = None


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration and return a mutable plain dictionary."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text)
    else:  # A useful fallback for the simple YAML shipped in this repository.
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return data


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed Python and NumPy and return an independent generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def freeze(value: Any) -> Any:
    """Convert nested observations into deterministic hashable keys."""
    if isinstance(value, dict):
        return tuple(sorted((k, freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    if isinstance(value, np.ndarray):
        return tuple(freeze(v) for v in value.tolist())
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def ensure_parent(path: str | Path) -> Path:
    """Create a file's parent directory and return the normalized path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def deep_get(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Read a nested config value without a chain of fragile ``get`` calls."""
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def flatten_dict(mapping: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dictionaries, useful for serializing config metadata."""
    result: Dict[str, Any] = {}
    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(flatten_dict(value, full_key))
        else:
            result[full_key] = value
    return result
