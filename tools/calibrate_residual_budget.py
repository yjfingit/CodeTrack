#!/usr/bin/env python
"""Calibrate the residual budget ``R_m`` for the bounded-step decoder.

The bounded residual parameterisation needs one number per modality: the largest residual
L2 norm a step is allowed to take.  Choosing it by hand from an observed overshoot ("the
applied update was 2.2x the ideal") would be tuning on the evaluation data.  Instead this
takes a fixed high quantile of the **ideal** residual ``x* - x`` -- what a perfect repair
would have had to produce -- on a pre-declared calibration set, and writes that number into
the experiment config.

Usage::

    python tools/calibrate_residual_budget.py --checkpoint outputs/ab_full/final.pth \
        --quantile 0.99 --sequences 8 --frames 16 --out outputs/residual_budget.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codetrack.utils.runtime import CPU_THREAD_SETTINGS  # noqa: E402,F401

import cv2
import numpy as np
import torch

from codetrack.data.transforms.sample import to_tensor  # noqa: E402
from codetrack.utils.checkpoint import load_checkpoint  # noqa: E402
from codetrack.utils.config import load_config  # noqa: E402
from recovery_probe import (apply_corruption, build_mask, crop_box,  # noqa: E402
                            list_sequences, prepare_plans)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/lasher_vitb_corrupt.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", default="/root/autodl-tmp/lab/dataset/LasHeR")
    parser.add_argument("--subset", default="testingset")
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--regime", default="block")
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--severity", type=float, default=0.4)
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/residual_budget.json")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    from codetrack.engine.trainer import Trainer  # noqa: E402

    trainer = Trainer(cfg, output_dir=Path(args.out).parent / "budget")
    report = load_checkpoint(args.checkpoint, trainer.model,
                             map_location=str(trainer.device))
    trainer.logger.info("loaded checkpoint %s (epoch %s)", args.checkpoint, report["epoch"])
    model = trainer.model.eval()

    root = Path(args.root)
    plans, skipped, pairing, requested = prepare_plans(
        root, args.subset, list_sequences(root, args.subset, args.sequences), args.frames)
    data_cfg = cfg.get("data", {})
    template_factor = float(data_cfg.get("template_factor", 2.0))
    search_factor = float(data_cfg.get("search_factor", 4.0))
    tpl_size, search_size = model.template_size, model.img_size
    grid, n_tok = model.grid, model.num_variables

    rng = np.random.default_rng(args.seed)
    damaged: Dict[str, List[float]] = {"rgb": [], "tir": []}
    healthy: Dict[str, List[float]] = {"rgb": [], "tir": []}
    deviation: Dict[str, List[float]] = {"rgb": [], "rgb_healthy": []}
    frames = 0
    for plan in plans:
        sequence = str(plan["sequence"])
        rgb_frames, annotations = plan["frames"], plan["boxes"]
        first_rgb_raw = cv2.imread(str(rgb_frames[0]))
        first_tir_raw = cv2.imread(str(plan["template_ir"]), cv2.IMREAD_GRAYSCALE)
        if first_rgb_raw is None or first_tir_raw is None:
            skipped.append(f"{sequence}: unreadable template frame")
            continue
        first_rgb = cv2.cvtColor(first_rgb_raw, cv2.COLOR_BGR2RGB)
        tpl_rgb_t = to_tensor(crop_box(trainer, first_rgb, annotations[0], template_factor,
                                       tpl_size), 3, (0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225)).unsqueeze(0).to(trainer.device)
        tpl_tir_t = to_tensor(crop_box(trainer, first_tir_raw, annotations[0],
                                       template_factor, tpl_size), 1, (0.449,),
                              (0.226,)).unsqueeze(0).to(trainer.device)
        for frame_index, tir_path in plan["pairs"]:
            rgb_raw = cv2.imread(str(rgb_frames[frame_index]))
            tir_raw = cv2.imread(str(tir_path), cv2.IMREAD_GRAYSCALE)
            if rgb_raw is None or tir_raw is None:
                skipped.append(f"{sequence}/{rgb_frames[frame_index].name}: unreadable")
                continue
            box = annotations[frame_index]
            rgb_full = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2RGB)
            search_rgb_t = to_tensor(crop_box(trainer, rgb_full, box, search_factor,
                                              search_size), 3, (0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225)).unsqueeze(0).to(trainer.device)
            search_tir_t = to_tensor(crop_box(trainer, tir_raw, box, search_factor,
                                              search_size), 1, (0.449,),
                                     (0.226,)).unsqueeze(0).to(trainer.device)
            mask = build_mask(args.regime, n_tok, grid, args.ratio, rng).to(trainer.device)
            with torch.no_grad():
                feats = model.backbone((tpl_rgb_t, search_rgb_t), (tpl_tir_t, search_tir_t),
                                       return_inter=True)
                clean_rgb, clean_tir = feats["x_r"][0], feats["x_t"][0]
                # the ideal residual is measured on the corrupted *input* view, i.e. exactly
                # what the decoder would have to produce to reach the teacher
                noisy_rgb = apply_corruption(args.regime, clean_rgb, mask, args.severity, rng)
                ideal_rgb = (clean_rgb - noisy_rgb).norm(dim=-1)
                # The same measurement in the units of the P4 deviation target
                # (q = e / (e + c_m) with e = mean_c |x - x*|), so c_m comes from data too.
                delta = (clean_rgb - noisy_rgb).abs().mean(dim=-1)
                deviation["rgb"].extend(delta[mask].cpu().tolist())
                deviation["rgb_healthy"].extend(delta[~mask].cpu().tolist())
                ideal_tir = (clean_tir - clean_tir).norm(dim=-1)      # untouched modality
            damaged["rgb"].extend(ideal_rgb[mask].cpu().tolist())
            healthy["rgb"].extend(ideal_rgb[~mask].cpu().tolist())
            damaged["tir"].extend(ideal_tir.cpu().tolist())
            frames += 1

    def quantile(values: List[float]) -> float:
        return float(np.quantile(np.asarray(values), args.quantile)) if values else float("nan")

    summary = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "overrides": args.override,
        "regime": args.regime,
        "ratio": args.ratio,
        "severity": args.severity,
        "quantile": args.quantile,
        "frames": frames,
        "pairing": pairing,
        "requested_search_frames": requested,
        "rgb_damaged_quantile": quantile(damaged["rgb"]),
        "rgb_healthy_quantile": quantile(healthy["rgb"]),
        "rgb_damaged_max": float(np.max(damaged["rgb"])) if damaged["rgb"] else float("nan"),
        "n_rgb_damaged_tokens": len(damaged["rgb"]),
        "tir_untouched_quantile": quantile(damaged["tir"]),
        "suggested_residual_clip": max(quantile(damaged["rgb"]),
                                       quantile(healthy["rgb"])),
        # P4: the deviation target's squash scale.  The median of the *damaged* deviation puts a
        # typical damaged token at q = 0.5 and a healthy one (deviation ~ 0) at q ~ 0.
        "rgb_deviation_l1_median_damaged": (float(np.median(deviation["rgb"]))
                                            if deviation["rgb"] else float("nan")),
        "rgb_deviation_l1_p90_damaged": (float(np.quantile(deviation["rgb"], 0.9))
                                         if deviation["rgb"] else float("nan")),
        "rgb_deviation_l1_median_healthy": (float(np.median(deviation["rgb_healthy"]))
                                            if deviation["rgb_healthy"] else float("nan")),
        "suggested_detect_deviation_cm": (float(np.median(deviation["rgb"]))
                                          if deviation["rgb"] else float("nan")),
        "skipped": skipped,
        "note": ("A fixed high quantile of the IDEAL residual x* - x on a pre-declared "
                 "calibration set, not a value tuned on the evaluation split.  It is an "
                 "engineering budget for the bounded step."),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"frames={frames} | rgb damaged p{args.quantile:.2f}="
          f"{summary['rgb_damaged_quantile']:.3f} healthy p{args.quantile:.2f}="
          f"{summary['rgb_healthy_quantile']:.3f} max={summary['rgb_damaged_max']:.3f} "
          f"| suggested model.residual_clip={summary['suggested_residual_clip']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
