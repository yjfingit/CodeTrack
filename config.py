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


def _parse_scalar(raw: str) -> Any:
    """Parse one override value.

    ``yaml.safe_load`` follows YAML 1.1, where ``off`` / ``on`` / ``yes`` / ``no`` are
    **booleans**.  So ``--override model.decoder_mode=off`` silently became ``False`` and the
    decoder fell through to the belief-propagation branch with ``parity_proj = None`` -- an
    ablation arm that crashed for a reason that had nothing to do with the ablation.

    Booleans are therefore handled here rather than by the YAML loader, and only the exact
    lowercase words that YAML 1.1 claims are mapped.
    """
    text = raw.strip()
    low = text.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~", ""):
        return None
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        # a bare string that YAML cannot read as anything else (e.g. "off" for a mode name)
        return text


def load_config(path: str | Path, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Load a yaml config, resolving ``_base_`` chains and applying overrides.

    ``overrides`` are ``key.sub=value`` strings; the value is parsed by :func:`_parse_scalar`,
    which does not treat ``off`` as a boolean.
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
        value = _parse_scalar(raw)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    cfg["_config_path"] = str(path)
    return cfg
