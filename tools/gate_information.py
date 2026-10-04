#!/usr/bin/env python
"""Do the gate inputs actually carry corruption information?

`gate_probe.py` measures what happens *after* the gate; this measures whether the gate and
its inputs rank corrupted tokens above healthy ones at all.  It matters because the whole
correction story depends on some signal being able to tell damage from health, and because
the detection heads are trained with the synthetic mask as the label -- which makes a high
AUROC partly circular unless it is reported as "predicts the injected mask", not as
"detects real degradation".

Per frame (RGB-only token corruption) it reports the ROC AUC of four scores against the
injected mask:

* ``reliability_rgb``      -- supervised with ``BCE(r, 1 - mask)``;
* ``locator_scattered``    -- ``H^T s``, supervised with ``BCE(locator, mask)``;
* ``syndrome``             -- the per-check map, scored per *check* against the realized
  corruption density of that check's neighbourhood (not per token);
* ``gate_rgb``             -- the severity gate handed to the decoder, scattered the way
  ``_expand_gate`` does (unselected variables keep the neutral 1).  This is *not* the
  applied coefficient;
* ``applied_coefficient``   -- ``(1 - r) * gate`` on the same grid, i.e. what the decoder
  actually multiplies the update by.  The two differ through the reliability factor and the
  neutral entries, and the selector's preference for damaged tokens makes the unscaled
  ``gate_rgb`` AUC uninterpretable on its own.

Usage::

    python tools/gate_information.py --checkpoint outputs/ab_full/final.pth \
        --sequences 2 --frames 8 --ratio 0.2 --out outputs/gate_information_ab_full.json
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
from scipy import stats

from codetrack.data.transforms.sample import to_tensor  # noqa: E402
from codetrack.engine.evaluator import auroc  # noqa: E402
from codetrack.utils.checkpoint import load_checkpoint  # noqa: E402
from codetrack.utils.config import load_config  # noqa: E402
from gate_probe import forced_gate  # noqa: E402
from recovery_probe import (apply_corruption, build_mask, crop_box,  # noqa: E402
                            list_sequences, prepare_plans)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/lasher_vitb_corrupt.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", default="/root/autodl-tmp/lab/dataset/LasHeR")
    parser.add_argument("--subset", default="testingset")
    parser.add_argument("--sequences", type=int, default=2)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--regime", default="block")
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--severity", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/gate_information.json")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    from codetrack.engine.trainer import Trainer  # noqa: E402

    trainer = Trainer(cfg, output_dir=Path(args.out).parent / "gateinfo")
    report = load_checkpoint(args.checkpoint, trainer.model,
                             map_location=str(trainer.device))
    trainer.logger.info("loaded checkpoint %s (epoch %s)", args.checkpoint, report["epoch"])
    model = trainer.model.eval()
    if model.decoder.mode == "off":
        raise SystemExit("decoder_mode=off: there is no gate to interrogate")

    root = Path(args.root)
    plans, skipped, pairing, requested = prepare_plans(
        root, args.subset, list_sequences(root, args.subset, args.sequences), args.frames)
    grid, n_tok = model.grid, model.num_variables
    data_cfg = cfg.get("data", {})
    template_factor = float(data_cfg.get("template_factor", 2.0))
    search_factor = float(data_cfg.get("search_factor", 4.0))
    tpl_size, search_size = model.template_size, model.img_size

    rng = np.random.default_rng(args.seed)
    scores: Dict[str, List[float]] = {"unreliability_rgb": [], "locator_scattered": [],
                                      "selection_of_damage": [], "gate_within_selected": [],
                                      "effective_gate_rgb": [], "applied_coefficient": [],
                                      "syndrome_density_rho": []}
    frames_done = 0
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
                noisy = apply_corruption(args.regime, clean_rgb, mask, args.severity, rng)
                with forced_gate(model, "learned", None) as captured:
                    out = model._code_path(
                        feats["z_r"], noisy.unsqueeze(0), feats["z_t"],
                        clean_tir.unsqueeze(0),
                        {"inter_r": feats["inter_r"], "inter_t": feats["inter_t"]})

            labels = mask.cpu().numpy().astype(np.int64)
            # reliability is a *health* score, so the informative reading is 1 - r
            scores["unreliability_rgb"].append(auroc(
                1.0 - out["reliability_rgb"][0].float().cpu().numpy(), labels))
            scores["locator_scattered"].append(auroc(
                out["locator_scattered"][0].float().cpu().numpy(), labels))

            # Gate components on the full variable grid.  ``effective_gate_rgb`` is the
            # network's *severity* gate only (selected nodes get the learned value,
            # unselected keep the neutral 1) -- it is NOT the coefficient the decoder
            # multiplies the update by.  That coefficient is ``(1 - r) * gate`` and is
            # scored separately as ``applied_coefficient``; on injected damage (1 - r) is 1,
            # so the two differ mainly through the reliability factor and the neutral
            # entries.  Reading ``effective_gate_rgb`` as "the decoder suppresses the
            # repair" was wrong: the selector prefers damaged tokens, so the neutral 1s are
            # concentrated on healthy tokens by construction.
            gate = captured.get("gate_rgb")
            index = captured.get("node_index")
            if gate is not None and index is not None:
                nodes = index[0].long().cpu()
                selected = torch.zeros(n_tok, dtype=torch.bool)
                selected[nodes] = True
                effective = torch.ones(n_tok, dtype=torch.float32)
                effective[nodes] = gate[0].float().cpu()
                selected_labels = labels[nodes.numpy()]
                scores["effective_gate_rgb"].append(auroc(effective.numpy(), labels))
                scores["selection_of_damage"].append(
                    auroc(selected.float().numpy(), labels))
                if 0 < selected_labels.sum() < len(selected_labels):
                    scores["gate_within_selected"].append(auroc(
                        gate[0].float().cpu().numpy(), selected_labels))
                reliability = out["reliability_rgb"][0].float().cpu()
                applied = (1.0 - reliability) * effective
                scores["applied_coefficient"].append(auroc(applied.numpy(), labels))

            # per-check syndrome vs the realized corruption density of that check.  The
            # evaluator reports a rank correlation, so a binary "density > 0" label is
            # useless here: with >=32 edges per check almost every check has density > 0.
            support = out["H_support"].float().cpu()
            degree = support.sum(dim=1).clamp(min=1.0)
            density = ((support @ mask.float().cpu()) / degree).numpy()
            check_scores = out["syndrome"].flatten(1)[0].float().cpu().numpy()
            if float(density.std()) > 1e-9 and float(np.std(check_scores)) > 1e-12:
                rank = stats.spearmanr(check_scores, density).statistic
                if np.isfinite(rank):
                    scores["syndrome_density_rho"].append(float(rank))
            frames_done += 1

    summary = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "overrides": args.override,
        "regime": args.regime,
        "ratio": args.ratio,
        "severity": args.seed,
        "frames": frames_done,
        "pairing": pairing,
        "requested_search_frames": requested,
        "auroc_mean": {key: float(np.nanmean(values)) if values else float("nan")
                       for key, values in scores.items()},
        "auroc_per_frame": {key: [float(value) for value in values]
                            for key, values in scores.items()},
        "skipped": skipped,
        "note": ("reliability/locator/syndrome are supervised with the injected mask, so a "
                 "high AUROC means 'predicts the synthetic mask', not 'detects real "
                 "degradation'; gate_rgb is not supervised with the mask at all."),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"frames={frames_done} " + "  ".join(
        f"{key}={value:.3f}" for key, value in summary["auroc_mean"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
