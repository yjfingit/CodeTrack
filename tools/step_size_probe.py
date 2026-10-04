#!/usr/bin/env python
"""Is the update uniformly too large, or do specific tokens deserve to be rejected?

`gate_probe.py` shows *what* the gate does; this separates the two competing explanations
without retraining, on one frozen frame at a time:

1. **uniform overshoot** -- the update direction `d = (1 - r) * u` is useful for most damaged
   tokens but the step is too large, so a single well-chosen step size would help most of them;
2. **selective rejection** -- for many damaged tokens even the *optimal* non-negative step
   `a* = clip(<d, e> / ||d||^2, 0, a_max)` is zero, i.e. the direction itself is unhelpful and
   no global rescaling can fix it.

Per damaged token it reports `a*` (oracle, offline only -- never a deployment result), the
step the model actually took (`||applied|| / ||d||`), the L1 error at zero / applied / oracle
step, and whether the learned gate predicts `a*` better than chance.  The gate is also permuted
*within strata* of (reliability, ||d||) so that only the gate's value information is destroyed
while its correlates survive; if the permutation changes nothing, the gate is decoration.

Usage::

    python tools/step_size_probe.py --checkpoint outputs/ab_full/final.pth \
        --sequences 4 --frames 12 --ratio 0.2 --out outputs/step_size_probe_ab_full.json
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
from codetrack.utils.checkpoint import load_checkpoint  # noqa: E402
from codetrack.utils.config import load_config  # noqa: E402
from gate_probe import forced_gate  # noqa: E402
from recovery_probe import (apply_corruption, build_mask, crop_box,  # noqa: E402
                            list_sequences, prepare_plans)

STEP_GRID = np.linspace(0.0, 2.0, 41)


def l1_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred - target).abs().mean())


def strata_keys(reliability: np.ndarray, norm: np.ndarray) -> np.ndarray:
    """Coarse (reliability, update-norm) strata so a permutation only destroys gate values."""
    key = np.zeros_like(reliability, dtype=np.int64)
    for values, weight in ((reliability, 2), (norm, 1)):
        ranks = stats.rankdata(values, method="average")
        buckets = np.clip((ranks / max(len(values), 1) * 4).astype(np.int64), 0, 3)
        key = key * 4 + buckets
    return key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/lasher_vitb_corrupt.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", default="/root/autodl-tmp/lab/dataset/LasHeR")
    parser.add_argument("--subset", default="testingset")
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--regime", default="block")
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--severity", type=float, default=0.4)
    parser.add_argument("--a-max", type=float, default=2.0)
    parser.add_argument("--single-step", action="store_true",
                        help="force model.bp_iterations=1.  With two iterations the gate "
                             "changes v after the first step, so the gate=1 and gate=learned "
                             "forwards no longer differ by a fixed per-token factor; the "
                             "single-step run is the clean reading of the oracle step")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/step_size_probe.json")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    overrides = list(args.override)
    if args.single_step:
        overrides.append("model.bp_iterations=1")

    cfg = load_config(args.config, overrides)
    from codetrack.engine.trainer import Trainer  # noqa: E402

    trainer = Trainer(cfg, output_dir=Path(args.out).parent / "stepprobe")
    report = load_checkpoint(args.checkpoint, trainer.model,
                             map_location=str(trainer.device))
    trainer.logger.info("loaded checkpoint %s (epoch %s)", args.checkpoint, report["epoch"])
    model = trainer.model.eval()
    if model.decoder.mode == "off":
        raise SystemExit("decoder_mode=off: there is no update to measure")
    out_norm = getattr(model.decoder, "out_norm", None)

    root = Path(args.root)
    plans, skipped, pairing, requested = prepare_plans(
        root, args.subset, list_sequences(root, args.subset, args.sequences), args.frames)
    data_cfg = cfg.get("data", {})
    template_factor = float(data_cfg.get("template_factor", 2.0))
    search_factor = float(data_cfg.get("search_factor", 4.0))
    tpl_size, search_size = model.template_size, model.img_size
    grid, n_tok = model.grid, model.num_variables

    rng = np.random.default_rng(args.seed)
    per_frame: List[Dict[str, float]] = []
    pooled = {key: [] for key in ("astar", "applied_ratio", "applied_over_astar", "gate_value",
                                 "reliability", "d_norm", "e_norm", "l1_zero", "l1_applied",
                                 "l1_oracle", "l1_best_global", "l1_gate_permuted",
                                 "l1_grid_best", "l1_oracle_postnorm", "l1_applied_postnorm")}
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
                inter = {"inter_r": feats["inter_r"], "inter_t": feats["inter_t"]}
                with forced_gate(model, "learned", None) as cap:
                    out_l = model._code_path(feats["z_r"], noisy.unsqueeze(0), feats["z_t"],
                                             clean_tir.unsqueeze(0), inter)
                with forced_gate(model, "full", None):
                    out_f = model._code_path(feats["z_r"], noisy.unsqueeze(0), feats["z_t"],
                                             clean_tir.unsqueeze(0), inter)

            damaged = mask.bool()
            if int(damaged.sum()) < 2:
                continue
            e = (clean_rgb - noisy)[damaged]                     # ideal update
            d = (out_f["pre_norm_rgb"][0] - noisy)[damaged]       # gate removed
            applied = (out_l["pre_norm_rgb"][0] - noisy)[damaged]
            reliability = out_l["reliability_rgb"][0][damaged]

            numerator = (d * e).sum(dim=-1)
            denominator = d.square().sum(dim=-1).clamp(min=1e-12)
            astar = (numerator / denominator).clamp(min=0.0, max=args.a_max)
            d_norm = d.norm(dim=-1)

            # The multiplier the decoder actually applies is the *captured* learned gate,
            # scattered exactly as ``_expand_gate`` does (unselected variables keep 1).  The
            # ratio ||applied|| / ||d|| is NOT that multiplier: with two BP iterations the
            # gate changes v after the first step, so the second iteration sees different
            # messages in the two forwards.  Use --single-step for the clean reading.
            learned_gate = cap.get("gate_rgb")
            nodes = cap.get("node_index")
            gate_value = torch.ones(damaged.shape[0], device=damaged.device)
            if learned_gate is not None and nodes is not None:
                full_grid = torch.ones(n_tok, device=damaged.device)
                full_grid[nodes[0].long()] = learned_gate[0].float()
                gate_value = full_grid[damaged]
            applied_ratio = applied.norm(dim=-1) / d_norm.clamp(min=1e-8)
            # The multiplier the decoder actually applied *along d*, and its ratio to the oracle
            # multiplier.  This -- not ``applied_ratio`` above -- is the quantity an "overshoot"
            # claim is about: ``||applied|| / ||d||`` coincides with it only under --single-step,
            # and ``L1(applied)/L1(oracle)`` (what tools/compare_step_probes.py prints) is an
            # error ratio again.  Both were read as "the step is 34x too large", which is why the
            # three are now reported next to each other with distinct names.
            a_applied = (applied * d).sum(dim=-1) / denominator
            ratio_valid = astar > 1e-6
            applied_over_astar = torch.where(
                ratio_valid, a_applied / astar.clamp(min=1e-12),
                torch.full_like(astar, float("nan")))
            # diagnostics of the gate's own distribution: the median over *all* damaged
            # tokens mixes selected (learned value) and unselected (neutral 1) entries
            selected_mask = torch.zeros(n_tok, dtype=torch.bool, device=damaged.device)
            if nodes is not None:
                selected_mask[nodes[0].long()] = True
            selected_damaged = selected_mask[damaged]
            gate_selected_mean = (float(gate_value[selected_damaged].mean())
                                  if bool(selected_damaged.any()) else float("nan"))

            zero = noisy[damaged]
            l1_zero = l1_error(zero, clean_rgb[damaged])
            l1_applied = l1_error(zero + applied, clean_rgb[damaged])
            l1_oracle = l1_error(zero + astar.unsqueeze(-1) * d, clean_rgb[damaged])

            # global single step size (what "everything is uniformly too big" predicts)
            grid_errors = [l1_error(zero + step * d, clean_rgb[damaged]) for step in STEP_GRID]
            best_index = int(np.argmin(grid_errors))
            l1_grid_best = float(grid_errors[best_index])
            best_step = float(STEP_GRID[best_index])

            # gate permutation within (reliability, norm) strata: only the values move
            keys = strata_keys(reliability.cpu().numpy(), d_norm.cpu().numpy())
            permuted = gate_value.clone()
            for key in np.unique(keys):
                index = torch.from_numpy(np.nonzero(keys == key)[0]).to(gate_value.device)
                if index.numel() > 1:
                    permuted[index] = gate_value[index][torch.randperm(index.numel(),
                                                                      device=gate_value.device)]
            l1_gate_permuted = l1_error(zero + permuted.unsqueeze(-1) * d, clean_rgb[damaged])

            l1_oracle_post = l1_applied_post = float("nan")
            if out_norm is not None:
                l1_oracle_post = l1_error(out_norm(zero + astar.unsqueeze(-1) * d),
                                          out_norm(clean_rgb[damaged]))
                l1_applied_post = l1_error(out_norm(zero + applied),
                                           out_norm(clean_rgb[damaged]))

            finite = torch.isfinite(astar)
            row = {
                "sequence": sequence,
                "frame": int(frame_index),
                "n_damaged": int(damaged.sum()),
                "astar_median": float(astar[finite].median()),
                "astar_zero_fraction": float((astar[finite] <= 1e-6).float().mean()),
                "applied_ratio_median": float(applied_ratio.median()),
                "a_applied_median": float(a_applied.median()),
                "applied_over_astar_median": float(torch.nanmedian(applied_over_astar)),
                "applied_over_astar_valid_fraction": float(ratio_valid.float().mean()),
                "gate_median": float(gate_value.median()),
                "gate_selected_mean": gate_selected_mean,
                "selected_fraction_damaged": float(selected_damaged.float().mean()),
                "gate_reliability_rho": float(stats.spearmanr(
                    gate_value.cpu().numpy(), astar.cpu().numpy()).statistic),
                "l1_zero": l1_zero, "l1_applied": l1_applied, "l1_oracle": l1_oracle,
                "l1_grid_best": l1_grid_best, "best_step": best_step,
                "l1_gate_permuted": l1_gate_permuted,
                "l1_oracle_postnorm": l1_oracle_post,
                "l1_applied_postnorm": l1_applied_post,
            }
            per_frame.append(row)
            pooled["astar"].append(astar[finite].cpu().numpy())
            pooled["applied_ratio"].append(applied_ratio.cpu().numpy())
            pooled["applied_over_astar"].append(applied_over_astar.cpu().numpy())
            pooled["gate_value"].append(gate_value.cpu().numpy())
            pooled["reliability"].append(reliability.cpu().numpy())
            pooled["d_norm"].append(d_norm.cpu().numpy())
            pooled["e_norm"].append(e.norm(dim=-1).cpu().numpy())
            for key in ("l1_zero", "l1_applied", "l1_oracle", "l1_best_global",
                        "l1_gate_permuted", "l1_oracle_postnorm", "l1_applied_postnorm"):
                if key == "l1_best_global":
                    pooled[key].append(np.array([l1_grid_best]))
                else:
                    pooled[key].append(np.array([row[key]]))
            frames_done += 1

    def stack(key: str) -> np.ndarray:
        values = [v for v in pooled[key] if np.size(v)]
        return np.concatenate(values) if values else np.zeros(0)

    astar_all = stack("astar")
    gate_all = stack("gate_value")
    summary = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "overrides": overrides,
        "single_step": bool(args.single_step),
        "regime": args.regime,
        "ratio": args.ratio,
        "severity": args.severity,
        "frames": frames_done,
        "pairing": pairing,
        "n_damaged_tokens": int(astar_all.size),
        "a_max": args.a_max,
        "astar_median": float(np.median(astar_all)) if astar_all.size else float("nan"),
        "astar_quartiles": [float(v) for v in np.quantile(astar_all, [0.25, 0.5, 0.75])]
        if astar_all.size else [],
        "astar_zero_fraction": float(np.mean(astar_all <= 1e-6)) if astar_all.size else float("nan"),
        "applied_ratio_to_d_median": float(np.median(stack("applied_ratio")))
        if stack("applied_ratio").size else float("nan"),
        # The three numbers that used to be conflated: ||applied||/||d|| (a step, but not the
        # step along d under multi-round BP), a_applied/a* (the actual overshoot factor), and
        # L1(applied)/L1(oracle) (an error ratio, reported by compare_step_probes).
        "applied_over_astar_median": float(np.nanmedian(stack("applied_over_astar")))
        if stack("applied_over_astar").size else float("nan"),
        "applied_over_astar_quartiles": [
            float(v) for v in np.nanquantile(stack("applied_over_astar"), [0.25, 0.5, 0.75])]
        if stack("applied_over_astar").size else [],
        "gate_median": float(np.median(gate_all)) if gate_all.size else float("nan"),
        "gate_selected_mean": float(np.nanmean([r["gate_selected_mean"] for r in per_frame]))
        if per_frame else float("nan"),
        "selected_fraction_damaged": float(np.mean([r["selected_fraction_damaged"]
                                                    for r in per_frame])) if per_frame
        else float("nan"),
        "gate_astar_spearman": float(stats.spearmanr(gate_all, astar_all).statistic)
        if astar_all.size > 2 else float("nan"),
        "mean_l1": {key: float(np.mean(stack(key))) if stack(key).size else float("nan")
                    for key in ("l1_zero", "l1_applied", "l1_oracle", "l1_best_global",
                                "l1_gate_permuted", "l1_oracle_postnorm",
                                "l1_applied_postnorm")},
        "per_frame": per_frame,
        "skipped": skipped,
        "note": ("a* is an offline oracle computed from the clean teacher; it is a diagnostic, "
                 "never a deployment result.  gate permutation is within (reliability, ||d||) "
                 "strata, so only the gate's value information is destroyed."),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    mean_l1 = summary["mean_l1"]
    print(f"frames={frames_done} damaged_tokens={summary['n_damaged_tokens']} "
          f"a* median={summary['astar_median']:.3f} "
          f"(zero fraction {summary['astar_zero_fraction']:.3f}) | "
          f"gate median={summary['gate_median']:.3f} "
          f"(selected mean {summary['gate_selected_mean']:.3f}, "
          f"selected frac {summary['selected_fraction_damaged']:.3f}) "
          f"(applied/||d|| median={summary['applied_ratio_to_d_median']:.3f}, "
          f"a_applied/a* median={summary['applied_over_astar_median']:.3f}) | "
          f"rho(gate, a*)={summary['gate_astar_spearman']:+.3f}")
    print(f"L1 on damaged tokens: zero={mean_l1['l1_zero']:.4f} "
          f"applied={mean_l1['l1_applied']:.4f} oracle={mean_l1['l1_oracle']:.4f} "
          f"best-global={mean_l1['l1_best_global']:.4f} "
          f"gate-permuted={mean_l1['l1_gate_permuted']:.4f} | "
          f"L1(applied)/L1(oracle)={mean_l1['l1_applied'] / mean_l1['l1_oracle']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
