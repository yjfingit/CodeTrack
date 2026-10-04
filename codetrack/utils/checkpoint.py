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
    """Load a checkpoint into ``model``; returns a short report.

    A checkpoint carries the resolved config it was trained with, and the decoder *mode*
    changes which parameters exist at all: loading an ``off`` arm into a model built for
    ``bp`` used to leave 16 message-passing tensors at their random initialisation, silently,
    because ``strict=False`` only reports the keys in a list nobody read.  Evaluation then
    measured a randomly wired decoder.  A mode mismatch is therefore an error; missing or
    unexpected keys are returned (and logged by the caller) so a shape drift cannot hide.
    """
    ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    stored = (ckpt.get("config") or {}) if isinstance(ckpt, dict) else {}
    stored_model = stored.get("model", {}) if isinstance(stored, dict) else {}
    live = getattr(model, "cfg", None) or {}
    live_model = live.get("model", {}) if isinstance(live, dict) else {}
    stored_mode = stored_model.get("decoder_mode")
    live_mode = live_model.get("decoder_mode", "bp")
    if stored_mode and stored_mode != live_mode:
        raise RuntimeError(
            f"{path} was trained with model.decoder_mode={stored_mode!r} but the model is "
            f"built with {live_mode!r}. Load it with "
            f"--override model.decoder_mode={stored_mode} (or build the matching config); "
            "loading it as-is leaves that arm's parameters randomly initialised."
        )

    # The same guard for the two settings that change the decoder *at run time* rather than in the
    # state dict.  ``decoder_output`` decides whether the branch norm and the residual projection
    # exist at all, so loading an identity-residual arm into a post_norm model silently drops those
    # tensors and evaluates a different network; ``gate_always_one`` decides whether the learned
    # severity gate is applied at all, and an arm trained with the gate forced to 1 never received
    # a gradient on that module, so evaluating it without the flag applies untrained weights.
    # Both mistakes happened while running the P2 grid (docs/results.md 6.23), so they now fail
    # loudly instead of producing a plausible-looking number.
    checks = (
        ("decoder_output", getattr(getattr(model, "decoder", None), "output_mode", None)),
        ("gate_always_one", getattr(getattr(model, "decoder", None), "gate_always_one", None)),
    )
    for key, live_value in checks:
        stored_value = stored_model.get(key)
        if stored_value is None or live_value is None:
            continue          # older checkpoints predate the key
        if str(stored_value).lower().lstrip("'") != str(live_value).lower().lstrip("'"):
            raise RuntimeError(
                f"{path} was trained with model.{key}={stored_value} but the model is built "
                f"with {live_value}.  Load it with --override model.{key}={stored_value} "
                "(the evaluation runner has an OVERRIDES variable for exactly this)."
            )

    missing, unexpected = model.load_state_dict(state, strict=strict)
    if optimizer is not None and isinstance(ckpt, dict) and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return {"epoch": ckpt.get("epoch", 0) if isinstance(ckpt, dict) else 0,
            "missing": list(missing), "unexpected": list(unexpected)}
