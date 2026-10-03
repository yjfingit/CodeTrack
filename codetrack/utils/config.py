"""Yaml config loading with ``_base_`` inheritance and dot-path overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Load a yaml config, resolving ``_base_`` chains and applying overrides.

    ``overrides`` are ``key.sub=value`` strings parsed with ``yaml.safe_load``.
    """
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    base_ref = cfg.pop("_base_", None)
    if base_ref:
        base_path = (path.parent / base_ref).resolve()
        cfg = _deep_merge(load_config(base_path), cfg)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    cfg["_config_path"] = str(path)
    return cfg
