"""Checkpoint save/load, including resolved config and RNG state for resumable runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


def save_checkpoint(path: Union[str, Path], model, optimizer=None, epoch: int = 0,
                    cfg: Optional[Dict[str, Any]] = None,
                    extra: Optional[Dict[str, Any]] = None,
                    best: bool = False) -> Path:
    """Write ``model`` (and optionally optimizer / config) to ``path``."""
    path = Path(path)
    if best:
        path = path.with_name(f"{path.stem}_best{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {"model": model.state_dict(), "epoch": epoch}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if cfg is not None:
        payload["config"] = cfg
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))
    return path


def load_checkpoint(path: Union[str, Path], model, optimizer=None,
                    map_location: str = "cpu", strict: bool = False) -> Dict[str, Any]:
    """Load a checkpoint into ``model``; returns a short report."""
    ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if optimizer is not None and isinstance(ckpt, dict) and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return {"epoch": ckpt.get("epoch", 0) if isinstance(ckpt, dict) else 0,
            "missing": list(missing), "unexpected": list(unexpected)}
