"""Tracking head construction and OSTrack head-weight loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import torch.nn as nn

from .center import CenterPredictor, conv
from .corner import CornerPredictor

_HEAD_PREFIXES = ("box_head.", "head.", "module.box_head.")


def build_head(cfg: Dict[str, Any]) -> nn.Module:
    """Build the tracking head.

    ``type`` is ``CENTER`` (default, matches the released OSTrack checkpoints) or
    ``CORNER``.  Both are drawn in the architecture figure.
    """
    kind = str(cfg.get("type", "CENTER")).upper()
    kwargs = dict(
        inplanes=int(cfg.get("inplanes", 768)),
        channel=int(cfg.get("channel", 256)),
        feat_sz=int(cfg.get("feat_sz", 16)),
        stride=int(cfg.get("stride", 16)),
    )
    if kind == "CENTER":
        return CenterPredictor(**kwargs)
    if kind in ("CORNER", "CORNER_PREDICTOR"):
        return CornerPredictor(**kwargs)
    raise ValueError(f"unsupported head type {cfg.get('type')!r}")


def _strip(key: str, prefixes: Iterable[str]) -> str:
    for p in prefixes:
        if key.startswith(p):
            return key[len(p):]
    return key


def load_head_weights(head: nn.Module, ckpt_path: str | Path,
                      verbose: bool = True) -> Dict[str, Any]:
    """Load the ``box_head.*`` tensors of an OSTrack checkpoint into ``head``.

    Only keys whose name **and** shape match are loaded; everything else is left at
    its initialisation.  Returns a short report.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt
    if isinstance(ckpt, dict):
        for k in ("net", "model", "state_dict"):
            if isinstance(ckpt.get(k), dict):
                state = ckpt[k]
                break

    own = head.state_dict()
    loadable: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        k = _strip(key, _HEAD_PREFIXES)
        if k in own and own[k].shape == value.shape:
            loadable[k] = value

    missing, unexpected = head.load_state_dict(loadable, strict=False)
    report = {"loaded": len(loadable), "total": len(own),
              "missing": [k for k in missing if k in own]}
    if verbose:
        print(f"[head] loaded {len(loadable)}/{len(own)} tensors from {Path(ckpt_path).name}")
        if report["missing"]:
            print(f"[head] kept initialised: {report['missing'][:6]}")
    return report
