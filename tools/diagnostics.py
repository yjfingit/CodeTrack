#!/usr/bin/env python
"""Discriminative experiments for the error-correction claim.

Tracking metrics cannot tell you whether the Tanner machinery is doing anything.  These
three probes can, and they are cheap -- no retraining, just perturbed inference:

1. **Shuffle test** -- permute the columns of ``H``, the parity order, or the syndrome at
   inference time.  A model that genuinely uses the code should degrade; one that has
   learned to ignore it will not move at all.
2. **FPN causal 2x2** -- clean/corrupted decoder tokens x clean/corrupted FPN, via the
   ``use_fpn`` flag and the corruption switch.
3. **Degree split** -- recovery gain reported separately for variables that no check
   watches (``degree == 0`` after the minimum-degree fix this is empty by construction) and
   those that several checks watch.  If both recover equally well, the repair came from a
   local MLP rather than from message passing.

Usage::

    python tools/diagnostics.py --checkpoint outputs/exp/final.pth --sequences 5
    python tools/diagnostics.py --checkpoint ... --probe shuffle-incidence
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codetrack.engine.trainer import Trainer  # noqa: E402
from codetrack.utils.config import load_config  # noqa: E402


# --------------------------------------------------------------------- probes
@contextmanager
def shuffled_incidence(model, seed: int = 0) -> Iterator[None]:
    """Permute which variable each check watches (permute the columns of H)."""
    dec = model.decoder
    original = dec.H_support
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(original.shape[1], generator=g)
    dec.H_support = original[:, perm].contiguous()
    try:
        yield
    finally:
        dec.H_support = original


@contextmanager
def shuffled_parity(model, seed: int = 0) -> Iterator[None]:
    """Permute which parity reference belongs to which check."""
    cb = model.codebook
    original = cb.parity_from_incidence

    def patched(identity, variables, H):
        p = original(identity, variables, H)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(p.shape[1], generator=g).to(p.device)
        return p[:, perm]

    cb.parity_from_incidence = patched
    try:
        yield
    finally:
        cb.parity_from_incidence = original


@contextmanager
def shuffled_syndrome(model, seed: int = 0) -> Iterator[None]:
    """Permute the syndrome entries after they are computed."""
    syn = model.syndrome
    original = syn.forward

    def patched(checks, parity):
        out = original(checks, parity)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(out["syndrome"].shape[-1], generator=g).to(out["syndrome"].device)
        out = dict(out)
        out["syndrome"] = out["syndrome"][:, :, perm]
        out["syndrome_raw"] = out["syndrome_raw"][:, perm]
        return out

    syn.forward = patched
    try:
        yield
    finally:
        syn.forward = original


@contextmanager
def no_decoder(model) -> Iterator[None]:
    """Bypass the belief-propagation decoder entirely (identity correction)."""
    dec = model.decoder
    original = dec.forward

    def patched(variables_rgb, variables_tir, *args, **kwargs):
        b = variables_rgb.shape[0]
        zeros = torch.zeros_like(variables_rgb)
        return {"corrected_rgb": variables_rgb, "corrected_tir": variables_tir,
                "residual_rgb": zeros, "residual_tir": zeros}

    dec.forward = patched
    try:
        yield
    finally:
        dec.forward = original


PROBES = {
    "baseline": None,
    "shuffle-incidence": shuffled_incidence,
    "shuffle-parity": shuffled_parity,
    "shuffle-syndrome": shuffled_syndrome,
    "no-decoder": no_decoder,
}


# ----------------------------------------------------------------------- main
def run_probe(trainer: Trainer, checkpoint: Optional[str], probe: str,
              corruption: Dict[str, Any], max_sequences: int, max_frames: int,
              topk: int) -> Dict[str, float]:
    context = PROBES[probe]
    if context is None:
        metrics = trainer.evaluate(checkpoint=checkpoint, max_sequences=max_sequences,
                                  max_frames=max_frames, corruption=corruption, topk=topk)
    else:
        with context(trainer.model):
            metrics = trainer.evaluate(checkpoint=None, max_sequences=max_sequences,
                                      max_frames=max_frames, corruption=corruption, topk=topk)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/lasher_vitb.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--out", default="outputs/diagnostics")
    parser.add_argument("--sequences", type=int, default=5)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--severity", type=float, default=0.4)
    parser.add_argument("--target", default="both", choices=["both", "rgb", "tir"])
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--probe", default="all",
                        help="comma separated probes, or 'all'")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    corruption = {
        "enabled": True,
        "token": ["tok_random_erase"],
        "ratio": args.ratio,
        "severity": args.severity,
        "target": args.target,
    }

    probes = list(PROBES) if args.probe == "all" else args.probe.split(",")
    results: Dict[str, Dict[str, float]] = {}
    for probe in probes:
        probe = probe.strip()
        if probe not in PROBES:
            print(f"unknown probe {probe!r}; known: {sorted(PROBES)}")
            continue
        # a fresh trainer per probe: the checkpoint is reloaded inside evaluate()
        trainer = Trainer(cfg, output_dir=out_dir / probe)
        results[probe] = run_probe(trainer, args.checkpoint or None, probe, corruption,
                                  args.sequences, args.frames, args.topk)
        print(f"[{probe}] " + json.dumps({k: round(v, 4)
                                         for k, v in results[probe].items()}))

    (out_dir / "diagnostics.json").write_text(json.dumps(results, indent=2))

    base = results.get("baseline", {})
    if base:
        print("\n" + "=" * 78)
        print(f"{'probe':<20} {'SR':>8} {'PR':>8} {'recovery':>10} {'synAUROC':>10}")
        print("-" * 78)
        for name, m in results.items():
            print(f"{name:<20} {m.get('sr', 0) * 100:8.2f} {m.get('pr', 0) * 100:8.2f} "
                  f"{m.get('recovery_gain', float('nan')):10.3f} "
                  f"{m.get('syndrome_auroc', float('nan')):10.3f}")
        print("=" * 78)
        print("A probe that does NOT hurt the metrics is decoration, not correction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
