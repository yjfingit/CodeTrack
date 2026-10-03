#!/usr/bin/env python
"""Shape audit for CodeTrack.

Runs one forward pass with random inputs (no dataset required) and compares every
intermediate tensor with the shape annotated on the architecture figure.  Also
reports how many OSTrack tensors were loaded into the backbone and the head.

Usage
-----
    python tools/check_shapes.py
    python tools/check_shapes.py --ckpt checkpoints/pretrained/OSTrack_ep0300.pth.tar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codetrack.models.codetrack import CodeTrack  # noqa: E402

#: tensor name -> (figure annotation, expected shape with batch 1)
EXPECTED = {
    "z_r": ("B x 64 x 768", (1, 64, 768)),
    "x_r": ("B x 256 x 768", (1, 256, 768)),
    "z_t": ("B x 64 x 768", (1, 64, 768)),
    "x_t": ("B x 256 x 768", (1, 256, 768)),
    "identity_tokens": ("Identity Tokens U  B x 16 x 256", (1, 16, 256)),
    "parity_tokens": ("Sparse Parity Tokens P  B x 16 x 256", (1, 16, 256)),
    "reliability_rgb": ("RGB query / reliability  B x 256", (1, 256)),
    "reliability_tir": ("TIR query / reliability  B x 256", (1, 256)),
    "A_uv": ("A_uv = Softmax_s(UU^T/sqrt(d))  B x 128 x 128", (1, 128, 128)),
    "identity_map": ("Identity Map  B x 128", (1, 128)),
    "syndrome": ("Syndrome S  B x 1 x 16", (1, 1, 16)),
    "locator": ("Error Locator  B x 128 x 256", (1, 128, 256)),
    "severity": ("Error Severity  B x 128", (1, 128)),
    "gate": ("Reliability-Aware Gating  B x 128", (1, 128)),
    "corrected_rgb": ("Corrected zg  B x 256 x 768", (1, 256, 768)),
    "corrected_tir": ("Corrected xg  B x 256 x 768", (1, 256, 768)),
    "residual_rgb": ("Residual  B x 256 x 768", (1, 256, 768)),
    "feature_map": ("Reshape to Feature Map  B x 768 x 16 x 16", (1, 768, 16, 16)),
    "fpn_rgb": ("FPN fusion RGB  B x 512 x 16 x 16", (1, 512, 16, 16)),
    "fpn_tir": ("FPN fusion TIR  B x 512 x 16 x 16", (1, 512, 16, 16)),
    "bbox": ("Tracking Head Bounding Box  B x 4", (1, 4)),
}


def default_model_cfg() -> dict:
    return {"model": {
        "backbone": "vitb_shared",
        "img_size": 256, "template_size": 128, "patch_size": 16,
        "embed_dim": 768, "depth": 12, "num_heads": 12,
        "num_identity_tokens": 16, "num_parity_tokens": 16,
        "num_variable_nodes": 256, "num_graph_nodes": 128,
        "code_dim": 256, "check_dim": 128, "fpn_dim": 256,
        "bp_iterations": 2, "return_stages": (2, 5),
        "freeze_backbone": True, "head_type": "CENTER",
    }}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="", help="OSTrack checkpoint to load")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-head", action="store_true")
    args = parser.parse_args()

    cfg = default_model_cfg()
    model = CodeTrack(cfg).to(args.device)

    if args.ckpt:
        ckpt = Path(args.ckpt)
        if ckpt.is_file():
            model.load_ostrack(str(ckpt), load_head=not args.no_head)
        else:
            print(f"!! checkpoint not found: {ckpt} (continuing with random init)")

    model.eval()
    b = args.batch
    rgb_t = torch.randn(b, 3, cfg["model"]["template_size"], cfg["model"]["template_size"],
                        device=args.device)
    rgb_s = torch.randn(b, 3, cfg["model"]["img_size"], cfg["model"]["img_size"],
                        device=args.device)
    tir_t = torch.randn(b, 1, cfg["model"]["template_size"], cfg["model"]["template_size"],
                        device=args.device)
    tir_s = torch.randn(b, 1, cfg["model"]["img_size"], cfg["model"]["img_size"],
                        device=args.device)

    with torch.no_grad():
        backbone_out = model.backbone((rgb_t, rgb_s), (tir_t, tir_s))
        out = model(rgb_t, rgb_s, tir_t, tir_s)

    shapes = dict(backbone_out)
    shapes.update(out)

    print(f"\n{'tensor':<18} {'figure annotation':<45} {'actual':<22} result")
    print("-" * 100)
    failures = []
    for name, (annotation, expected) in EXPECTED.items():
        if name not in shapes:
            print(f"{name:<18} {annotation:<45} {'MISSING':<22} FAIL")
            failures.append(name)
            continue
        actual = tuple(shapes[name].shape)
        ok = actual == expected if b == 1 else actual[0] == b and actual[1:] == expected[1:]
        print(f"{name:<18} {annotation:<45} {str(actual):<22} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(name)

    print("\n--- parameter summary (trainable / total) ---")
    total_t = total_a = 0
    for name, (tr, tot) in model.parameter_summary().items():
        total_t += tr
        total_a += tot
        print(f"  {name:<12} {tr/1e6:9.4f} M / {tot/1e6:9.4f} M")
    print(f"  {'TOTAL':<12} {total_t/1e6:9.4f} M / {total_a/1e6:9.4f} M")

    print(f"\nshape audit: {'PASS' if not failures else 'FAIL -> ' + ', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
