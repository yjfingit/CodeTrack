"""Provenance for every run: which code, which environment, which live configuration.

Why this exists
---------------
The 073c223 round showed the cost of *not* recording this.  ``docs/results.md`` 6.10 attributed
arm A's mismatch with the shipped ``ab_full`` checkpoint to a shifted RNG stream; the real cause
was a change in the decoder loop's state semantics, and nothing in the artifacts said which code
had produced which number.  A result that cannot be tied to a commit cannot be re-derived, and a
comparison whose two arms were evaluated by different code is not a comparison.

Everything here is best-effort: a missing ``git`` binary or a tarball checkout degrades to
``"unknown"`` rather than raising, because failing a training run over provenance would just
encourage removing the call.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def _git(args: list, cwd: Path) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True,
                                timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def git_provenance(cwd: Path = Path(".")) -> Dict[str, Any]:
    """Commit, dirty flag and branch for ``cwd``.

    ``dirty`` matters more than the commit: a dirty tree means the number cannot be reproduced
    from the commit alone, which is exactly the situation the provenance audit has to detect.
    """
    commit = _git(["rev-parse", "HEAD"], cwd)
    status = _git(["status", "--porcelain"], cwd)
    return {
        "git_commit": commit,
        "git_short": commit[:7] if commit != "unknown" else "unknown",
        "git_dirty": None if status == "unknown" else bool(status),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd),
    }


def model_provenance(model: Any) -> Dict[str, Any]:
    """The run-time switches that change the decoder's *function* without changing its tensors.

    These are the settings that made an evaluated network differ from the trained one, so they
    belong next to every number rather than only in a log line.
    """
    decoder = getattr(model, "decoder", None)
    out: Dict[str, Any] = {
        "decoder_mode": getattr(decoder, "mode", None),
        "decoder_output": getattr(decoder, "output_mode", None),
        "decoder_state_mode": getattr(decoder, "state_mode", None),
        "gate_always_one": getattr(decoder, "gate_always_one", None),
        "residual_clip": getattr(decoder, "residual_clip", None),
        "bp_iterations": getattr(decoder, "iterations", None),
    }
    backbone = getattr(model, "backbone", None)
    if backbone is not None:
        out["backbone_frozen"] = getattr(backbone, "frozen", None)
    return out


def parameter_provenance(model: Any) -> Dict[str, Any]:
    """How many parameters actually receive gradients -- the number a fairness claim needs.

    ``total`` counts every tensor; ``active`` counts only parameters with ``requires_grad``,
    which is the quantity that must be matched between two arms for "equal capacity" to mean
    anything.
    """
    total = active = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            active += count
    return {"parameters_total": total, "parameters_active": active}


def peak_memory_mb() -> Dict[str, Any]:
    """Peak CUDA memory in MiB, so a budget claim is measured rather than asserted."""
    try:
        import torch
    except ImportError:
        return {"peak_memory_mb": None}
    if not torch.cuda.is_available():
        return {"peak_memory_mb": None}
    return {"peak_memory_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)}


def run_provenance(cwd: Path = Path("."), model: Any = None,
                   extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Everything above, as one dict for ``run_provenance.json`` / ``run_manifest.json``."""
    payload: Dict[str, Any] = {
        **git_provenance(cwd),
        "cwd": str(Path(cwd).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_thread_env": {k: v for k, v in os.environ.items()
                           if k.endswith("_NUM_THREADS") or k.startswith("CODETRACK_")},
    }
    if model is not None:
        payload.update(model_provenance(model))
        payload.update(parameter_provenance(model))
    payload.update(peak_memory_mb())
    if extra:
        payload.update(extra)
    return payload
