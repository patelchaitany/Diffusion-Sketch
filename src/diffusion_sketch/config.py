"""YAML-based configuration with CLI override support.

Usage:
    cfg = load_config()                          # loads configs/default.yaml
    cfg = load_config("configs/custom.yaml")     # loads a custom file
    cfg = load_config(overrides={"training.epochs": 50})  # with overrides
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using 'a.b.c' notation."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


class Config(dict):
    """Dict subclass with attribute access (cfg.training.epochs)."""

    def __getattr__(self, name: str):
        try:
            val = self[name]
        except KeyError:
            raise AttributeError(f"Config has no key '{name}'")
        return Config(val) if isinstance(val, dict) else val

    def __setattr__(self, name: str, value: Any):
        self[name] = value


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load YAML config, optionally merging CLI overrides.

    Args:
        path: path to YAML file. Falls back to configs/default.yaml.
        overrides: dict of dotted-key overrides, e.g. {"training.epochs": 50}.
    """
    cfg_path = Path(path) if path else _DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path) as f:
        raw = yaml.safe_load(f) or {}

    if overrides:
        for key, val in overrides.items():
            _set_nested(raw, key, val)

    return Config(raw)
