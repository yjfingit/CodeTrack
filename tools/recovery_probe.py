#!/usr/bin/env python
"""Recovery, measured on real LasHeR frames with real block corruption.

`syndrome_fit.py` feeds the model ``torch.randn`` images.  That is fine for a *relative*
comparison between corruption types -- the corruption is synthetic anyway -- but it is not a
validation, and a pearson measured on noise says nothing about natural imagery.  This tool
closes that gap and fixes the definition problems the review found:

* **real frames** from the dataset, so the token statistics are the ones the tracker sees;
* **four corruption regimes** -- 2-D block erase (what the locality prior is built for),
  energy-matched block *replacement* (fills with the neighbourhood's own mean, so the energy
  signature the obs_energy bypass exploited is gone), random erase (negative control), and
  additive feature noise;
* **norm-only control** -- the decoder's output LayerNorm applied to the uncorrected input.
  Neither the corrupted input nor the clean teacher passes through it, so without this the
  reported error mixes "the messages helped/hurt" with "the scale moved";
* **the identity G = 1 - E_after / E_before** on the same samples, same masks, same
  aggregation, plus the p95 of the per-frame ratio.

Usage::

    python tools/recovery_probe.py --checkpoint outputs/<exp>/final.pth --sequences 20
    python tools/recovery_probe.py --checkpoint ... --regimes block_replace random
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codetrack.data.transforms.sample import to_tensor          # noqa: E402
from codetrack.utils.config import load_config                     # noqa: E402
from codetrack.utils.checkpoint import load_checkpoint             # noqa: E402


def build_mask(kind: str, n: int, grid: int, ratio: float,
               rng: np.random.Generator) -> torch.Tensor:
    """``n``-token boolean mask in the order the backbone produces patches."""
    k = max(1, int(round(ratio * n)))
    mask = torch.zeros(n, dtype=torch.bool)
    if kind in ("block", "block_replace"):
        side = max(1, int(round(np.sqrt(k))))
        r0 = int(rng.integers(0, max(1, grid - side + 1)))
        c0 = int(rng.integers(0, max(1, grid - side + 1)))
        # a 2-D block on the search grid -- the 1-D flattened version the earlier burst
        # corruption used is exactly what the locality prior must not be matched against
        for dr in range(side):
            for dc in range(side):
                mask[(r0 + dr) * grid + (c0 + dc)] = True
    elif kind == "random":
        mask[torch.from_numpy(rng.permutation(n)[:k])] = True
    else:
        raise ValueError(kind)
    return mask


def apply_corruption(kind: str, tokens: torch.Tensor, mask: torch.Tensor,
                     severity: float, rng: np.random.Generator) -> torch.Tensor:
    """``tokens``: ``N x C``.  Returns the corrupted tokens."""
    if not bool(mask.any()):
        return tokens
    if kind == "random":
        return tokens.masked_fill(mask.unsqueeze(-1), 0.0)

    if kind == "block_replace":
        # energy-matched replacement: fill with the *healthy* neighbourhood's mean, so the
        # corrupted patch is statistically indistinguishable from an undamaged one.  This is
        # the regime that removes the local energy signature; if the syndrome still works
        # here, it is not reading energy.
        healthy = tokens[~mask]
        fill = healthy.mean(dim=0, keepdim=True) if healthy.numel() else tokens.mean(0, keepdim=True)
        out = tokens.clone()
        out[mask] = fill + 0.0 * severity
        return out
    return tokens.masked_fill(mask.unsqueeze(-1), 0.0)          # "block"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment/lasher_vitb_corrupt.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root", default="/root/autodl-tmp/lab/dataset/LasHeR")
    ap.add_argument("--subset", default="testingset")
    ap.add_argument("--sequences", type=int, default=20)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--ratio", type=float, default=0.2)
    ap.add_argument("--severity", type=float, default=0.4)
    ap.add_argument("--regimes", default="block,block_replace,random,feat_noise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/recovery_probe.json")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    from codetrack.engine.trainer import Trainer                     # noqa: E402
    trainer = Trainer(cfg, output_dir=Path(args.out).parent / "recprobe")
    load_checkpoint(args.checkpoint, trainer.model, map_location="cpu")
    model = trainer.model.to(trainer.device).eval()

    root = Path(args.root)
    list_name = "testingsetList.txt" if args.subset.startswith("test") else "trainingsetList.txt"
    seqs = [s.strip() for s in (root / list_name).read_text().splitlines() if s.strip()]
    seqs = seqs[: args.sequences]

    grid = model.grid
    n_tok = model.num_variables
    tpl_size = model.template_size
    search_size = model.img_size

    rows: List[Dict[str, float]] = []
    skipped: List[str] = []
    for regime in [r.strip() for r in args.regimes.split(",") if r.strip()]:
        rng = np.random.default_rng(args.seed)
        acc: Dict[str, List[float]] = {"e_before": [], "e_after": [], "e_norm": [],
                                       "g_frame": []}
        for seq in seqs:
            sdir = root / args.subset / seq
            # LasHeR stores visible/vNNN.jpg and infrared/iNNN.jpg -- same index, different
            # prefix, so they must be paired by number rather than by file name.
            vframes = sorted((sdir / "visible").glob("v*.jpg"))
            if len(vframes) < 2:
                skipped.append(f"{seq}: only {len(vframes)} visible frames")
                continue

            def infrared_for(vpath: Path) -> Optional[Path]:
                cand = sdir / "infrared" / ("i" + vpath.name[1:])
                return cand if cand.exists() else None

            first = vframes[0]
            ip = infrared_for(first)
            if ip is None:
                skipped.append(f"{seq}: no infrared for {first.name}")
                continue
            tpl_full = cv2.cvtColor(cv2.imread(str(first)), cv2.COLOR_BGR2RGB)
            tpl_t_full = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
            if tpl_full is None or tpl_t_full is None:
                skipped.append(f"{seq}: unreadable template frame")
                continue
            h0, w0 = tpl_full.shape[:2]
            cx0, cy0 = w0 / 2, h0 / 2
            half = tpl_size / 2
            if min(h0, w0) < tpl_size:
                skipped.append(f"{seq}: frame smaller than template {tpl_size}")
                continue
            tpl = tpl_full[int(cy0 - half):int(cy0 + half), int(cx0 - half):int(cx0 + half)]
            tpl_t = tpl_t_full[int(cy0 - half):int(cy0 + half), int(cx0 - half):int(cx0 + half)]
            tpl_rgb = to_tensor(tpl, 3, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            tpl_tir = to_tensor(tpl_t, 1, (0.449,), (0.226,))

            for f in vframes[1: 1 + args.frames]:
                ip = infrared_for(f)
                if ip is None:
                    skipped.append(f"{seq}/{f.name}: no infrared pair")
                    continue
                rgb_full = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
                tir_full = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
                if rgb_full is None or tir_full is None:
                    skipped.append(f"{seq}/{f.name}: unreadable")
                    continue
                # centre crop at a fixed scale so the template/search sizes match the model's
                # pos_embed; the frames are raw and have no such size
                h, w = rgb_full.shape[:2]
                cx, cy = w / 2, h / 2
                half_t, half_s = tpl_size / 2, search_size / 2

                def crop(img):
                    return img[int(cy - half_s):int(cy + half_s),
                               int(cx - half_s):int(cx + half_s)]

                rgb_s = crop(rgb_full)
                tir_s = crop(tir_full)
                if min(rgb_s.shape[:2]) < search_size or min(tir_s.shape[:2]) < search_size:
                    skipped.append(f"{seq}/{f.name}: frame smaller than {search_size}")
                    continue
                s_rgb = to_tensor(rgb_s, 3, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))[None]
                s_tir = to_tensor(tir_s, 1, (0.449,), (0.226,))[None]
                mask = build_mask("block" if regime == "feat_noise" else regime,
                                  n_tok, grid, args.ratio, rng)

                args4 = (tpl_rgb[None].to(trainer.device), s_rgb.to(trainer.device),
                         tpl_tir[None].to(trainer.device), s_tir.to(trainer.device))
                with torch.no_grad():
                    feats = model.backbone((args4[0], args4[1]), (args4[2], args4[3]),
                                           return_inter=True)
                    clean = feats["x_r"][0]
                    noisy = apply_corruption(regime, clean, mask.to(clean.device),
                                             args.severity, rng)
                    if regime == "feat_noise":
                        noisy = noisy + torch.randn_like(noisy) * (
                            args.severity * clean.std())
                        mask = (noisy - clean).abs().sum(-1) > 0

                    class _Out(dict):
                        pass
                    # the FPN taps are not what this probe measures, but _code_path still
                    # runs them, so pass the real ones rather than an empty dict
                    out = model._code_path(
                        feats["z_r"], noisy[None], feats["z_t"], feats["x_t"][0][None],
                        {"inter_r": feats["inter_r"], "inter_t": feats["inter_t"]})
                    dev = clean.device
                    out["token_mask_rgb"] = mask[None].float().to(dev)
                    out["token_mask_tir"] = mask[None].float().to(dev)
                    out["corrupted_rgb"] = noisy[None]
                    out["corrupted_tir"] = feats["x_t"][0][None]
                    out["clean_tokens"] = {"rgb": clean[None], "tir": feats["x_t"][0][None]}

                    rec = trainer.loss_fn.masked_recovery(out)
                acc["e_before"].append(float(rec["e_before"].mean()))
                acc["e_after"].append(float(rec["e_after"].mean()))
                if "e_norm_only" in rec:
                    acc["e_norm"].append(float(rec["e_norm_only"].mean()))
                acc["g_frame"].append(float(1.0 - rec["ratio"].mean()))

        if not acc["e_before"]:
            print(f"[{regime}] no usable frames; skipped {len(skipped)} "
                  f"(first: {skipped[:2]})")
            continue
        eb, ea = float(np.mean(acc["e_before"])), float(np.mean(acc["e_after"]))
        row = {
            "regime": regime,
            "e_before": eb,
            "e_after": ea,
            "G": 1.0 - ea / max(eb, 1e-8),
            "G_frame_p05": float(np.quantile(acc["g_frame"], 0.05)),
            "G_frame_median": float(np.median(acc["g_frame"])),
            "G_frame_p95": float(np.quantile(acc["g_frame"], 0.95)),
            "frames": len(acc["e_before"]),
        }
        if acc["e_norm"]:
            en = float(np.mean(acc["e_norm"]))
            row["e_norm_only"] = en
            row["G_norm_only"] = 1.0 - en / max(eb, 1e-8)
        rows.append(row)

    print("\n=== recovery on real LasHeR frames (G = 1 - E_after/E_before, higher is better) ===")
    hdr = "%-15s %9s %9s %8s %9s %9s %9s" % (
        "regime", "e_before", "e_after", "G", "G_p05", "G_med", "G_p95")
    if rows and "e_norm_only" in rows[0]:
        hdr += " %10s %10s" % ("e_norm", "G_norm")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = "%-15s %9.4f %9.4f %8.3f %9.3f %9.3f %9.3f" % (
            r["regime"], r["e_before"], r["e_after"], r["G"],
            r["G_frame_p05"], r["G_frame_median"], r["G_frame_p95"])
        if "e_norm_only" in r:
            line += " %10.4f %10.3f" % (r["e_norm_only"], r["G_norm_only"])
        print(line)
    print("\nG > 0 means the correction moved the tokens toward the clean teacher.")
    print("G_norm is the same measure with the message passing removed: the decoder's output")
    print("LayerNorm alone.  Compare G against G_norm to see what the Tanner messages bought.")

    Path(args.out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
