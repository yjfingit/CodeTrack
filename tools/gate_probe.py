#!/usr/bin/env python
"""Interrogate the decoder's update direction and its severity gate, on real frames.

``recovery_probe.py`` answers "is the full decoder output better than the corrupted
input"; it cannot say whether a bad result comes from the update *direction* or from the
gate that scales it.  The four cells are re-measured here under four gate policies, from
one model forward and the same corruption sample per frame:

* ``learned`` -- the model's own severity gate;
* ``zero``    -- gate 0 everywhere; the pre-norm cell must then equal the corrupted input
  exactly (a falsifiable identity check, reported as ``identity_check``);
* ``full``    -- gate 1 on every selected node, i.e. the severity gate removed while the
  ``(1 - r)`` reliability factor still applies;
* ``oracle``  -- gate 1 exactly on the corrupted variables and 0 elsewhere, using the
  corruption mask the probe itself drew.

It also reports, per policy, the cosine between the applied update and the ideal update
``clean - corrupted`` on damaged tokens, plus the applied/ideal norm ratio.  Those two
numbers separate "the update points the wrong way" from "the update points the right way
but is scaled to nothing", which the L1 error alone cannot distinguish.

``learned_gate`` reports where the model actually puts its gate: the mean gate on selected
nodes that are corrupted, the mean on selected nodes that are healthy, and the fraction of
the 256 variables that the selected graph covers.

Usage::

    python tools/gate_probe.py --checkpoint outputs/x_full/final.pth --sequences 20 \
        --frames 12 --gates learned,zero,full,oracle --out outputs/gate_probe_x_full.json
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codetrack.utils.runtime import CPU_THREAD_SETTINGS  # noqa: E402,F401

import cv2
import numpy as np
import torch

from codetrack.data.transforms.sample import to_tensor  # noqa: E402
from codetrack.utils.checkpoint import load_checkpoint  # noqa: E402
from codetrack.utils.config import load_config  # noqa: E402
from recovery_probe import (apply_corruption, build_mask, crop_box,  # noqa: E402
                            list_sequences, prepare_plans, target_region_mask,
                            token_error)

POLICIES = ("learned", "zero", "full", "oracle")
CELLS = ("v", "ln_v", "v_plus_d", "ln_v_plus_d")
CATEGORIES = ("rgb_damaged", "rgb_healthy", "rgb_target", "tir_all", "tir_target",
              "both_target")


@contextmanager
def forced_gate(model, policy: str, mask: Optional[torch.Tensor]) -> Iterator[Dict]:
    """Run the decoder with a substituted severity gate, capturing what it received.

    The gate enters the decoder through ``gate_rgb`` / ``gate_tir``.  A ``B x K`` gate is
    scattered onto the selected nodes by ``_expand_gate`` and every *unselected* variable
    keeps the neutral multiplier ``1`` -- so zeroing a ``K``-shaped gate only silences half
    the grid.  ``zero`` and ``full`` therefore pass a full ``B x N`` gate, and ``oracle``
    does the same, which makes the intervention exact on the variable grid.
    """
    decoder = model.decoder
    original = decoder.forward
    captured: Dict[str, object] = {"policy": policy}

    def patched(*args, **kwargs):
        gate_rgb = kwargs.get("gate_rgb")
        gate_tir = kwargs.get("gate_tir")
        captured["node_index"] = kwargs.get("node_index")
        captured["gate_rgb"] = gate_rgb
        captured["gate_tir"] = gate_tir
        if policy in ("zero", "full", "oracle"):
            # The received gate is B x K on the *selected* nodes; the variable grid is the
            # first positional argument (B x N x d).  Building a K-wide replacement here
            # was the bug that made "gate 0" still move half the tokens.
            variables = kwargs.get("variables_rgb")
            if variables is None and args:
                variables = args[0]
            if not torch.is_tensor(variables):
                raise ValueError("cannot size the substituted gate without variable tokens")
            batch, num_variables = variables.shape[0], variables.shape[1]
            if policy == "oracle":
                if mask is None:
                    raise ValueError("the oracle gate needs the corruption mask")
                oracle = mask.to(device=gate_rgb.device, dtype=gate_rgb.dtype)
                kwargs["gate_rgb"] = oracle
                kwargs["gate_tir"] = torch.zeros_like(oracle)
            else:
                fill = 0.0 if policy == "zero" else 1.0
                full = torch.full((batch, num_variables), fill,
                                  device=gate_rgb.device, dtype=gate_rgb.dtype)
                kwargs["gate_rgb"] = full
                kwargs["gate_tir"] = full.clone()
        return original(*args, **kwargs)

    decoder.forward = patched
    try:
        yield captured
    finally:
        decoder.forward = original


def gate_statistics(captured: Dict, mask: torch.Tensor, n_variables: int) -> Dict[str, float]:
    """Where the learned gate lands, restricted to the selected graph nodes."""
    gate = captured.get("gate_rgb")
    node_index = captured.get("node_index")
    stats: Dict[str, float] = {"selected_fraction": float("nan")}
    if gate is None or node_index is None:
        return stats
    gate = gate.detach().float().flatten()
    index = node_index.detach().long().flatten()
    stats["selected_fraction"] = float(index.numel()) / float(n_variables)
    damaged = mask.to(device=index.device).flatten()[index].bool()
    if bool(damaged.any()):
        stats["damaged_mean"] = float(gate[damaged].mean())
    if bool((~damaged).any()):
        stats["healthy_mean"] = float(gate[~damaged].mean())
    stats["overall_mean"] = float(gate.mean())
    return stats


def update_direction(pred: torch.Tensor, corrupted: torch.Tensor, clean: torch.Tensor,
                     region: torch.Tensor) -> Dict[str, float]:
    """Cosine and norm ratio between the applied update and the ideal update."""
    applied = (pred - corrupted).float()
    ideal = (clean - corrupted).float()
    if not bool(region.any()):
        return {"delta_cos_damaged": float("nan"),
                "delta_norm_ratio_damaged": float("nan")}
    applied = applied[region]
    ideal = ideal[region]
    cos = torch.nn.functional.cosine_similarity(applied, ideal, dim=-1, eps=1e-8)
    applied_norm = applied.norm(dim=-1).mean()
    ideal_norm = ideal.norm(dim=-1).mean().clamp(min=1e-8)
    return {"delta_cos_damaged": float(cos.mean()),
            "delta_norm_ratio_damaged": float(applied_norm / ideal_norm)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/lasher_vitb_corrupt.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", default="/root/autodl-tmp/lab/dataset/LasHeR")
    parser.add_argument("--subset", default="testingset")
    parser.add_argument("--sequences", type=int, default=20)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--regimes", default="block,block_replace")
    parser.add_argument("--gates", default="learned,zero,full,oracle")
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--severity", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/gate_probe.json")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    from codetrack.engine.trainer import Trainer  # noqa: E402

    trainer = Trainer(cfg, output_dir=Path(args.out).parent / "gateprobe")
    report = load_checkpoint(args.checkpoint, trainer.model,
                             map_location=str(trainer.device))
    trainer.logger.info("loaded checkpoint %s (epoch %s)", args.checkpoint,
                        report["epoch"])
    model = trainer.model.eval()
    if model.decoder.mode == "off":
        raise SystemExit("decoder_mode=off: there is no update branch or gate to probe")

    policies: List[str] = [value.strip() for value in args.gates.split(",") if value.strip()]
    unknown = [value for value in policies if value not in POLICIES]
    if unknown:
        raise SystemExit(f"unknown gate policies {unknown}; known: {list(POLICIES)}")
    regimes = [value.strip() for value in args.regimes.split(",") if value.strip()]

    root = Path(args.root)
    sequences = list_sequences(root, args.subset, args.sequences)
    plans, skipped, pairing_counts, requested_frames = prepare_plans(
        root, args.subset, sequences, args.frames)

    grid = model.grid
    n_tok = model.num_variables
    tpl_size = model.template_size
    search_size = model.img_size
    data_cfg = cfg.get("data", {})
    template_factor = float(data_cfg.get("template_factor", 2.0))
    search_factor = float(data_cfg.get("search_factor", 4.0))

    rows: List[Dict[str, object]] = []
    learned_gate: Dict[str, Dict[str, float]] = {}

    for regime in regimes:
        rng = np.random.default_rng(args.seed)
        measurements = {policy: {cell: {category: [] for category in CATEGORIES}
                                 for cell in CELLS} for policy in policies}
        direction = {policy: {"delta_cos_damaged": [], "delta_norm_ratio_damaged": []}
                     for policy in policies}
        gate_stats: List[Dict[str, float]] = []
        identity_ok = True
        frames_done = 0

        for plan in plans:
            sequence = str(plan["sequence"])
            rgb_frames = plan["frames"]
            annotations = plan["boxes"]
            first_ir = plan["template_ir"]
            first_rgb_raw = cv2.imread(str(rgb_frames[0]))
            first_tir_raw = cv2.imread(str(first_ir), cv2.IMREAD_GRAYSCALE)
            if first_rgb_raw is None or first_tir_raw is None:
                skipped.append(f"{sequence}: unreadable template frame")
                continue
            first_rgb = cv2.cvtColor(first_rgb_raw, cv2.COLOR_BGR2RGB)
            template_rgb = crop_box(trainer, first_rgb, annotations[0], template_factor,
                                    tpl_size)
            template_tir = crop_box(trainer, first_tir_raw, annotations[0], template_factor,
                                    tpl_size)
            tpl_rgb_t = to_tensor(template_rgb, 3, (0.485, 0.456, 0.406),
                                  (0.229, 0.224, 0.225)).unsqueeze(0).to(trainer.device)
            tpl_tir_t = to_tensor(template_tir, 1, (0.449,), (0.226,)).unsqueeze(0).to(
                trainer.device)

            for frame_index, tir_path in plan["pairs"]:
                rgb_path = rgb_frames[frame_index]
                rgb_raw = cv2.imread(str(rgb_path))
                tir_raw = cv2.imread(str(tir_path), cv2.IMREAD_GRAYSCALE)
                if rgb_raw is None or tir_raw is None:
                    skipped.append(f"{sequence}/{rgb_path.name}: unreadable frame")
                    continue
                rgb_full = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2RGB)
                box = annotations[frame_index]
                search_rgb = crop_box(trainer, rgb_full, box, search_factor, search_size)
                search_tir = crop_box(trainer, tir_raw, box, search_factor, search_size)
                search_rgb_t = to_tensor(search_rgb, 3, (0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225)).unsqueeze(0).to(trainer.device)
                search_tir_t = to_tensor(search_tir, 1, (0.449,), (0.226,)).unsqueeze(0).to(
                    trainer.device)

                mask = build_mask(regime, n_tok, grid, args.ratio, rng).to(trainer.device)
                with torch.no_grad():
                    feats = model.backbone((tpl_rgb_t, search_rgb_t),
                                           (tpl_tir_t, search_tir_t), return_inter=True)
                    clean_rgb = feats["x_r"][0]
                    clean_tir = feats["x_t"][0]
                    noisy_rgb = apply_corruption(regime, clean_rgb, mask, args.severity, rng)

                    search_side = float(np.sqrt(max(float(box[2]), 1.0) *
                                                max(float(box[3]), 1.0))) * search_factor
                    box_w = float(box[2]) / search_side
                    box_h = float(box[3]) / search_side
                    target_mask = target_region_mask(
                        np.array([0.5 - box_w / 2.0, 0.5 - box_h / 2.0, box_w, box_h],
                                 dtype=np.float32), grid).to(trainer.device)

                    for policy in policies:
                        mask_bn = mask.unsqueeze(0).float() if policy == "oracle" else None
                        with forced_gate(model, policy, mask_bn) as captured:
                            out = model._code_path(
                                feats["z_r"], noisy_rgb.unsqueeze(0), feats["z_t"],
                                clean_tir.unsqueeze(0),
                                {"inter_r": feats["inter_r"], "inter_t": feats["inter_t"]})
                        if policy == "learned":
                            for key, value in gate_statistics(captured, mask, n_tok).items():
                                gate_stats.append({key: value})
                        if policy == "zero":
                            pre_norm = out["pre_norm_rgb"][0]
                            identity_ok = identity_ok and bool(torch.allclose(
                                pre_norm, noisy_rgb, atol=1e-6, rtol=1e-5))

                        views = {
                            "v": (noisy_rgb, clean_tir),
                            "ln_v": (out["norm_only_rgb"][0], out["norm_only_tir"][0]),
                            "v_plus_d": (out["pre_norm_rgb"][0], out["pre_norm_tir"][0]),
                            "ln_v_plus_d": (out["corrected_rgb"][0],
                                            out["corrected_tir"][0]),
                        }
                        for cell, (pred_rgb, pred_tir) in views.items():
                            values = (
                                token_error(pred_rgb, clean_rgb, mask),
                                token_error(pred_rgb, clean_rgb, ~mask),
                                token_error(pred_rgb, clean_rgb, target_mask),
                                token_error(pred_tir, clean_tir,
                                            torch.ones_like(mask, dtype=torch.bool)),
                                token_error(pred_tir, clean_tir, target_mask),
                                0.5 * (token_error(pred_rgb, clean_rgb, target_mask)
                                       + token_error(pred_tir, clean_tir, target_mask)),
                            )
                            for category, value in zip(CATEGORIES, values):
                                measurements[policy][cell][category].append(value)
                        for key, value in update_direction(
                                out["pre_norm_rgb"][0], noisy_rgb, clean_rgb, mask).items():
                            direction[policy][key].append(value)
                frames_done += 1

        for policy in policies:
            errors = {
                cell: {category: float(np.nanmean(values)) if values else float("nan")
                       for category, values in by_category.items()}
                for cell, by_category in measurements[policy].items()
            }
            damaged = errors["v_plus_d"]["rgb_damaged"]
            row = {
                "regime": regime,
                "policy": policy,
                "frames": frames_done,
                "errors": errors,
                "ratio": args.ratio,
                "severity": args.severity,
                "damaged_rgb_gain_pre_norm": 1.0 - damaged /
                max(errors["v"]["rgb_damaged"], 1e-8),
                "damaged_rgb_gain_post_norm": 1.0 - errors["ln_v_plus_d"]["rgb_damaged"] /
                max(errors["v"]["rgb_damaged"], 1e-8),
                "delta_cos_damaged": float(np.nanmean(direction[policy]
                                                      ["delta_cos_damaged"])),
                "delta_norm_ratio_damaged": float(np.nanmean(direction[policy]
                                                             ["delta_norm_ratio_damaged"])),
            }
            if policy == "zero":
                row["identity_check"] = bool(identity_ok)
            rows.append(row)
            print(f"[{regime}/{policy}] frames={frames_done} "
                  f"damaged RGB: v={errors['v']['rgb_damaged']:.4f} "
                  f"v+D={errors['v_plus_d']['rgb_damaged']:.4f} "
                  f"LN(v+D)={errors['ln_v_plus_d']['rgb_damaged']:.4f} | "
                  f"cos={row['delta_cos_damaged']:+.3f} "
                  f"|d|/|d*|={row['delta_norm_ratio_damaged']:.3f} | "
                  f"healthy damage={errors['ln_v_plus_d']['rgb_healthy']:.4f} "
                  f"TIR damage={errors['ln_v_plus_d']['tir_all']:.4f}")
        if gate_stats:
            keys = sorted({key for entry in gate_stats for key in entry})
            learned_gate[regime] = {
                key: float(np.nanmean([entry.get(key, float("nan"))
                                       for entry in gate_stats])) for key in keys}
            print(f"[{regime}] learned gate: " + ", ".join(
                f"{key}={learned_gate[regime][key]:.4f}" for key in keys))

    payload = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "overrides": args.override,
        "subset": args.subset,
        "requested_sequences": args.sequences,
        "usable_sequences": len(plans),
        "frames_per_sequence_cap": args.frames,
        "requested_search_frames": requested_frames,
        "measured_frames_per_regime": rows[0]["frames"] if rows else 0,
        "ratio": args.ratio,
        "severity": args.severity,
        "seed": args.seed,
        "regimes": regimes,
        "gates": policies,
        "corruption_target": "rgb",
        "pairing": pairing_counts,
        "learned_gate": learned_gate,
        "rows": rows,
        "skipped": skipped,
    }
    print("\n`v` and `ln_v` do not depend on the gate policy; they are recomputed under "
          "each policy as a consistency check.")
    print("`oracle` opens the gate exactly on the corrupted variables; `full` opens it "
          "everywhere but keeps (1 - r).")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
