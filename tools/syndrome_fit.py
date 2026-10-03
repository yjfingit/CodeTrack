#!/usr/bin/env python
"""Is the syndrome actually predicting the corruption density?

Fits ``S`` against ``q`` on a *fresh, unseen* corruption draw and reports

* ``pearson`` / ``spearman`` -- does S move with q at all?
* ``slope of a least-squares fit`` -- 1.0 means the scale is right too;
* ``mean S | q=0`` vs ``mean S | q>0.4`` -- the two ends must separate;
* ``soft-BCE`` against the constant predictor ``q.mean()`` -- the number training
  minimises.  If S cannot beat a *constant*, the term is decoration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codetrack.engine.trainer import Trainer                      # noqa: E402
from codetrack.utils.config import load_config                    # noqa: E402
from codetrack.engine.evaluator import spearman                   # noqa: E402


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def soft_bce(pred: np.ndarray, target: np.ndarray) -> float:
    p = np.clip(pred, 1e-6, 1 - 1e-6)
    t = np.clip(target, 0.0, 1.0)
    return float(-(t * np.log(p) + (1 - t) * np.log(1 - p)).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment/lasher_vitb_corrupt.yaml")
    ap.add_argument("--checkpoint", default="outputs/r3_train/last.pth")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--ratio", type=float, default=0.2)
    ap.add_argument("--severity", type=float, default=0.4)
    ap.add_argument("--token", default="tok_random_erase",
                    help="corruption to MEASURE with; should match what the checkpoint was "
                         "trained on, since the energy statistics differ per type and a "
                         "zero-erase measurement of a noise-trained model is meaningless")
    ap.add_argument("--energy", action="store_true",
                    help="set model.syndrome_use_obs_energy=true (must match training)")
    ap.add_argument("--out", default="outputs/syndrome_fit.json")
    ap.add_argument("--override", action="append", default=[],
                    help="config override key.sub=value, must match the checkpoint's shape")
    args = ap.parse_args()

    # The checkpoint fixes the architecture (M, N, degree); the default config would build
    # a differently-shaped model and load_state_dict would fail with an opaque message.
    override = list(args.override)
    if args.energy:
        override.append("model.syndrome_use_obs_energy=true")
    cfg = load_config(args.config, override)
    trainer = Trainer(cfg, output_dir=Path(args.out).parent / "synfit")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
    missing, unexpected = trainer.model.load_state_dict(state, strict=False)
    hard = [k for k in missing if "temperature_logit" not in k]
    if hard:
        raise SystemExit(
            f"checkpoint does not fit this architecture: {len(hard)} missing tensors, "
            f"e.g. {hard[:3]}.\nThe checkpoint was trained with different model settings "
            f"(num_parity_tokens / h_links_per_check / ...); pass the same values:\n"
            f"  --override model.num_parity_tokens=32"
        )
    model = trainer.model.to(trainer.device).eval()

    b, n = 16, model.num_variables
    support = model.decoder.connectivity().to(trainer.device)
    degree = support.sum(dim=1).clamp(min=1.0)

    preds, targets, clean_preds = [], [], []
    # upstream instrumentation: is the corruption visible in `checks` at all, and does the
    # reference swamp it?
    clean_checks, corrupt_checks = [], []
    delta_norms = []
    agg_norms, ref_norms = [], []
    with torch.no_grad():
        for _ in range(args.frames):
            tpl = torch.randn(b, 3, model.template_size, model.template_size,
                              device=trainer.device)
            srh = torch.randn(b, 3, model.img_size, model.img_size,
                              device=trainer.device)
            tpl_t = torch.randn(b, 1, model.template_size, model.template_size,
                                device=trainer.device)
            srh_t = torch.randn(b, 1, model.img_size, model.img_size,
                                device=trainer.device)

            out_a = model(tpl, srh, tpl_t, srh_t, corruption={
                "enabled": True, "token": [args.token],
                "ratio": args.ratio, "severity": args.severity, "target": "both"})

            # the label: density over the SAME support the model used
            q = ((support @ (out_a["token_mask_rgb"] + out_a["token_mask_tir"]).clamp(max=1).t())
                 / degree.unsqueeze(1)).t().clamp(0, 1)

            preds.append(out_a["syndrome"].flatten().float().cpu().numpy())
            targets.append(q.flatten().cpu().numpy())

            out_c = model(tpl, srh, tpl_t, srh_t, corruption=None)
            clean_preds.append(out_c["syndrome"].flatten().float().cpu().numpy())

            if "check_agg" in out_a:
                corrupt_checks.append(out_a["check_agg"].flatten().float().cpu().numpy())
                clean_checks.append(out_c["check_agg"].flatten().float().cpu().numpy())
                d = (out_a["check_agg"] - out_a["check_ref"])
                agg_norms.append(float(out_a["check_agg"].norm(dim=-1).mean()))
                ref_norms.append(float(out_a["check_ref"].norm(dim=-1).mean()))
                delta_norms.append(float(d.norm(dim=-1).mean()))

    cc, kc = np.concatenate(corrupt_checks), np.concatenate(clean_checks)
    # RMS difference between the corrupted and the clean check activations, relative to
    # their own scale: if this is ~0, the corruption is invisible before the head
    denom = max(float(np.abs(kc).mean()), 1e-8)
    clean_vs_corrupt_rel = float(np.sqrt(((cc - kc) ** 2).mean()) / denom)
    agg_norm = float(np.mean(agg_norms)) if agg_norms else float("nan")
    ref_norm = float(np.mean(ref_norms)) if ref_norms else float("nan")
    delta_norm = float(np.mean(delta_norms)) if delta_norms else float("nan")
    delta_over_agg = delta_norm / max(agg_norm, 1e-8)
    ref_over_agg = ref_norm / max(agg_norm, 1e-8)

    p = np.concatenate(preds)
    t = np.concatenate(targets)
    c = np.concatenate(clean_preds)

    slope, intercept = np.polyfit(t, p, 1)
    const = np.full_like(p, t.mean())
    result = {
        "n_values": int(p.size),
        "syndrome_mean": float(p.mean()),
        "syndrome_std": float(p.std()),
        "density_mean": float(t.mean()),
        "density_std": float(t.std()),
        "pearson": pearson(p, t),
        "spearman": spearman(p, t),
        "ls_slope": float(slope),
        "ls_intercept": float(intercept),
        "spearman_vs_clean": spearman(np.concatenate([p, c]),
                                      np.concatenate([t, np.zeros_like(c)])),
        "mean_S_clean_frames": float(c.mean()),
        "mean_S_q0": float(p[t == 0].mean()) if (t == 0).any() else float("nan"),
        "mean_S_q_gt_0p4": float(p[t > 0.4].mean()) if (t > 0.4).any() else float("nan"),
        "softbce_model": soft_bce(p, t),
        "softbce_constant_mean": soft_bce(const, t),
        "mae_model": float(np.abs(p - t).mean()),
        "mae_constant_mean": float(np.abs(const - t).mean()),
        # upstream: does the evidence even reach the head?
        "checks_clean_vs_corrupt_rel_diff": float(clean_vs_corrupt_rel),
        "delta_norm_over_agg_norm": float(delta_over_agg),
        "ref_norm_over_agg_norm": float(ref_over_agg),
    }
    print("\n=== syndrome vs corruption density ===")
    for k, v in result.items():
        print(f"  {k:<34} {v}")
    beats = result["softbce_model"] < result["softbce_constant_mean"] - 1e-4
    print(f"\n  model beats a constant predictor: {beats}")
    if not beats:
        print("  -> this probe did not show syndrome information above its constant baseline.")
        print("     Check the corruption, support degrees, and held-out runs before drawing a")
        print("     conclusion about whether the graph can use this corruption type.")
    Path(args.out).write_text(__import__("json").dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
