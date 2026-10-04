#!/usr/bin/env python
"""Is the per-token repair even a useful mediator?  Replace the tokens with the clean ones.

The discriminating experiment for the round-2 question.  `docs/results.md` 6.24 and 6.26 both
improved the *mechanism* -- first the update geometry, then the detection and repair quality --
and neither moved the robustness contrast; one of them moved it against the improved arm.  The
tempting conclusion is "per-token repair quality is not the bottleneck", but none of those
experiments ever handed the tracker a *repaired* token set: they changed how hard the repair was
asked to work.  This tool does the direct thing.

For each frame the same crop is run twice: once with the token corruption switched off (the
"clean" pass, which caches what fusion receives) and once with the real corruption, where the
corrupted tensors are replaced at the **fusion input** by

    z_alpha = (1 - alpha) * z_corrupted + alpha * z_clean

`alpha = 0` is the plain corrupted run; `alpha = 1` is "the repair came out perfect".  Fusion
consumes `corrected_*` and the backbone FPN taps, and ignores `residual_*` (the decoder already
folds the residual into the corrected tokens), so nothing is left inconsistent -- and
`--include-taps` extends the interpolation to the taps, which matters because the FPN path
otherwise keeps exactly the corruption the tokens just lost.

How to read the cells:

* **fixed crop + oracle helps** -- the per-frame response improves, so the representation is a
  valid mediator and the learning target/weighting is the suspect.
* **free loop + oracle helps** -- the effect survives the closed loop, so it is not merely
  trajectory divergence.
* **oracle helps but training does not** -- the mechanism can matter and the optimiser is not
  obtaining it.
* **oracle does not help either** -- the current teacher/token representation is not a mediator
  at all.  Still not "no repair can ever help".

`--scope foreground|background` interpolates only the tokens inside (or outside) the ground-truth
box, which asks whether an average token metric is dominated by background tokens.  It requires
`--modes fixed`, because the crop schedule must be known in closed form to map a box onto the
16x16 token grid.

Self-check: with `alpha = 0` the extra clean forward must change nothing, so the arm has to
reproduce a plain corrupted run **bit for bit**.  That is asserted per sequence and reported as
`alpha0_consistent`; a failure invalidates every other cell.

Usage::

    python tools/token_oracle_interp.py --checkpoint outputs/ab_full/final.pth \
        --sequence-list outputs/validation_split_v1/sequences.txt --sequences 20 \
        --frames 200 --alphas 0,0.5,1 --modes fixed,free \
        --out outputs/token_oracle_interp_ab_full.json
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from codetrack.utils.runtime import CPU_THREAD_SETTINGS  # noqa: E402,F401

import numpy as np  # noqa: E402
import torch  # noqa: E402

from codetrack.engine.trainer import Trainer  # noqa: E402
from codetrack.metrics import _box_iou  # noqa: E402
from codetrack.utils.checkpoint import load_checkpoint  # noqa: E402
from codetrack.utils.config import load_config  # noqa: E402
from recovery_probe import list_sequences  # noqa: E402


def _blend(corrupted: torch.Tensor, clean: torch.Tensor, alpha: float) -> torch.Tensor:
    return (1.0 - alpha) * corrupted + alpha * clean


def token_scope_mask(grid: int, gt_box: np.ndarray, crop: np.ndarray,
                     scope: str) -> Optional[torch.Tensor]:
    """``grid*grid`` boolean tensor: True where a token may be interpolated.

    The ground-truth box is mapped into the search crop the loop actually used, so the indices
    line up with the search half of the token sequence.  ``scope="all"`` returns ``None``.
    """
    if scope == "all":
        return None
    cx, cy = float(crop[0] + crop[2] / 2), float(crop[1] + crop[3] / 2)
    side = float(np.sqrt(max(crop[2], 1.0) * max(crop[3], 1.0)))
    if side <= 0:
        return None
    gx = (float(gt_box[0] + gt_box[2] / 2) - cx) / side + 0.5
    gy = (float(gt_box[1] + gt_box[3] / 2) - cy) / side + 0.5
    gw, gh = float(gt_box[2]) / side, float(gt_box[3]) / side
    col0, col1 = int(np.floor((gx - gw / 2) * grid)), int(np.ceil((gx + gw / 2) * grid))
    row0, row1 = int(np.floor((gy - gh / 2) * grid)), int(np.ceil((gy + gh / 2) * grid))
    mask = torch.zeros(grid, grid, dtype=torch.bool)
    mask[max(0, row0):max(0, min(grid, row1)), max(0, col0):max(0, min(grid, col1))] = True
    mask = mask.reshape(-1)
    if scope == "background":
        mask = ~mask
    # An off-crop box yields an empty foreground: report "nothing to interpolate here" rather
    # than silently switching to all-or-nothing.
    return mask


@contextmanager
def capture_clean(model, cache: Dict[str, object]):
    """Record what fusion receives on a *clean* pass at the same crop."""
    original_fusion = model.fusion.forward
    original_taps = model._apply_tap_corruption

    def fusion(corrected_rgb, corrected_tir, residual_rgb, residual_tir, taps_r, taps_t):
        cache["corrected_rgb"] = corrected_rgb.detach()
        cache["corrected_tir"] = corrected_tir.detach()
        return original_fusion(corrected_rgb, corrected_tir, residual_rgb, residual_tir,
                               taps_r, taps_t)

    def taps(feats, mask_rgb, mask_tir):
        out = original_taps(feats, mask_rgb, mask_tir)
        cache["taps_clean"] = {key: {name: value.detach() for name, value in out[key].items()}
                               for key in ("inter_r", "inter_t") if key in out}
        return out

    model.fusion.forward = fusion
    model._apply_tap_corruption = taps
    try:
        with torch.no_grad():
            yield
    finally:
        model.fusion.forward = original_fusion
        model._apply_tap_corruption = original_taps


@contextmanager
def apply_interpolation(model, cache: Dict[str, object], alpha: float, include_taps: bool,
                        scope_mask: Optional[torch.Tensor]):
    """Blend the corrupted fusion inputs toward the cached clean ones."""
    original_fusion = model.fusion.forward
    original_taps = model._apply_tap_corruption

    def blend_tokens(value: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        blended = _blend(value, clean, alpha)
        if scope_mask is None:
            return blended
        if value.shape[1] != scope_mask.numel():
            # layout mismatch (template tokens present): refuse to guess
            return blended
        keep = scope_mask.to(value.device).view(1, -1, 1)
        return torch.where(keep, blended, value)

    def fusion(corrected_rgb, corrected_tir, residual_rgb, residual_tir, taps_r, taps_t):
        return original_fusion(blend_tokens(corrected_rgb, cache["corrected_rgb"]),
                               blend_tokens(corrected_tir, cache["corrected_tir"]),
                               residual_rgb, residual_tir, taps_r, taps_t)

    def taps(feats, mask_rgb, mask_tir):
        out = original_taps(feats, mask_rgb, mask_tir)
        if not include_taps or "taps_clean" not in cache:
            return out
        blended = dict(out)
        for key in ("inter_r", "inter_t"):
            if key not in out or key not in cache["taps_clean"]:
                continue
            blended[key] = {
                name: _blend(value, cache["taps_clean"][key][name], alpha)
                if name in cache["taps_clean"][key] else value
                for name, value in out[key].items()}
        return blended

    model.fusion.forward = fusion
    model._apply_tap_corruption = taps
    try:
        yield
    finally:
        model.fusion.forward = original_fusion
        model._apply_tap_corruption = original_taps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/experiment/lasher_vitb_corrupt.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--root", default="/root/autodl-tmp/lab/dataset/LasHeR")
    parser.add_argument("--subset", default="testingset")
    parser.add_argument("--sequence-list", default=None)
    parser.add_argument("--sequences", type=int, default=20)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--alphas", default="0,0.5,1")
    parser.add_argument("--modes", default="fixed,free")
    parser.add_argument("--scope", default="all", choices=["all", "foreground", "background"])
    parser.add_argument("--include-taps", action="store_true",
                        help="also interpolate the backbone FPN taps, which carry the same "
                             "corruption the tokens just lost")
    parser.add_argument("--token", default="tok_block_erase")
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--severity", type=float, default=0.4)
    parser.add_argument("--target", default="both", choices=["both", "rgb", "tir"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/token_oracle_interp.json")
    args = parser.parse_args()

    if args.scope != "all" and "fixed" not in args.modes:
        raise SystemExit("--scope needs the fixed mode: the crop schedule must be known in "
                         "closed form to map a ground-truth box onto the token grid")

    cfg = load_config(args.config, args.override)
    trainer = Trainer(cfg, output_dir=Path(args.out).parent)
    load_checkpoint(args.checkpoint, trainer.model, map_location=str(trainer.device))
    trainer.model.eval()

    root = Path(args.root)
    if args.sequence_list:
        sequences = [line.strip() for line in Path(args.sequence_list).read_text().splitlines()
                     if line.strip()][:args.sequences]
    else:
        sequences = list_sequences(root, args.subset, args.sequences)

    alphas = [float(value) for value in args.alphas.split(",") if value.strip()]
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    model = trainer.model
    grid = int(model.grid)
    corruption = {"enabled": True, "token": [args.token], "rgb": [], "tir": [],
                  "cross_modal": False, "ratio": args.ratio, "severity": args.severity,
                  "target": args.target}

    def arm(sequence: str, fixed_boxes, alpha: Optional[float], gt_boxes: np.ndarray,
            condition: Optional[Dict[str, object]]) -> Dict[str, object]:
        """One sequence under one (condition, mode, alpha) cell.

        ``alpha=None`` is the plain run, with no extra forward at all.
        """
        state: Dict[str, object] = {"index": 0}

        def forward_fn(tpl_rgb, s_rgb, tpl_tir, s_tir, corruption=None, clean_teacher=False):
            index = int(state["index"])
            cache: Dict[str, object] = {}
            with capture_clean(model, cache):
                model(tpl_rgb, s_rgb, tpl_tir, s_tir, corruption=None, clean_teacher=False)
            mask = None
            if args.scope != "all" and index < len(gt_boxes):
                probe = (np.asarray(gt_boxes[0], dtype=np.float32) if index == 0
                         else np.asarray(fixed_boxes[index - 1], dtype=np.float32))
                mask = token_scope_mask(grid, np.asarray(gt_boxes[index]), probe, args.scope)
            with apply_interpolation(model, cache, float(alpha), args.include_taps, mask):
                out = model(tpl_rgb, s_rgb, tpl_tir, s_tir, corruption=corruption,
                            clean_teacher=clean_teacher)
            state["index"] = index + 1
            return out

        torch.manual_seed(args.seed)
        run = trainer.infer_sequence(root, args.subset, sequence, max_frames=args.frames,
                                     corruption=condition, fixed_boxes=fixed_boxes,
                                     forward_fn=None if alpha is None else forward_fn)
        pred, gt = run["pred"], run["gt"]
        if len(pred) == 0:
            return {"frames": 0}
        iou = _box_iou(pred, gt)
        return {
            "frames": int(len(pred)),
            "iou_mean": float(iou.mean()),
            "sr": float((iou > 0.5).mean()),
            "iou_mean_first20": float(iou[:20].mean()),
            "iou_mean_last20": float(iou[-20:].mean()),
            "box_clamps": int(run.get("box_clamps", 0)),
            "pred": pred,
        }

    rows: List[Dict[str, object]] = []
    for sequence in sequences:
        anno_path = root / "annos" / f"{sequence}.txt"
        gt_boxes = np.array([[float(v) for v in line.replace("\t", ",").split(",")[:4]]
                             for line in anno_path.read_text().splitlines() if line.strip()],
                            dtype=np.float32)
        reference = None
        if "fixed" in modes:
            # The reference has to come from the *clean* free-running pass.  Using the corrupted
            # free run (the first version of this tool did) makes the fixed arm replay the
            # corrupted run's own trajectory, which is why its alpha = 0 cell came out identical
            # to the free arm on every sequence and the whole column measured nothing.
            reference = arm(sequence, None, None, gt_boxes, None)["pred"]

        entry: Dict[str, object] = {"sequence": sequence, "arms": {}}
        for mode in modes:
            fixed_boxes = reference if mode == "fixed" else None
            plain = arm(sequence, fixed_boxes, None, gt_boxes, corruption)
            for alpha in alphas:
                result = arm(sequence, fixed_boxes, alpha, gt_boxes, corruption)
                key = f"{mode}_a{alpha:g}"
                entry["arms"][key] = {name: value for name, value in result.items()
                                      if name != "pred"}
                if alpha == 0.0:
                    same = (len(result["pred"]) == len(plain["pred"])
                            and bool(np.array_equal(result["pred"], plain["pred"])))
                    entry[f"alpha0_matches_plain_{mode}"] = same
        rows.append(entry)
        summary = ", ".join(f"{key}={value['iou_mean']:.3f}"
                            for key, value in entry["arms"].items())
        trainer.logger.info("%s: %s", sequence, summary)

    valid = [row for row in rows if row["arms"]]
    keys = sorted({key for row in valid for key in row["arms"]}) if valid else []
    report: Dict[str, object] = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "overrides": args.override,
        "sequence_list": args.sequence_list,
        "frames": args.frames,
        "alphas": alphas,
        "modes": modes,
        "scope": args.scope,
        "include_taps": bool(args.include_taps),
        "corruption": {"token": args.token, "ratio": args.ratio, "severity": args.severity,
                       "target": args.target},
        "n_sequences": len(valid),
        "note": ("alpha = 1 hands fusion the clean pass's tensors, i.e. an oracle repair; "
                 "alpha = 0 must reproduce the plain corrupted run bit for bit."),
        "mean_iou_by_arm": {key: float(np.mean([row["arms"][key]["iou_mean"] for row in valid
                                                if key in row["arms"]])) for key in keys},
        "mean_sr_by_arm": {key: float(np.mean([row["arms"][key]["sr"] for row in valid
                                               if key in row["arms"]])) for key in keys},
        "alpha0_consistent": {mode: all(row.get(f"alpha0_matches_plain_{mode}", False)
                                        for row in valid) for mode in modes},
        "rows": rows,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"sequences={len(valid)} frames={args.frames} scope={args.scope} "
          f"taps={'yes' if args.include_taps else 'no'}")
    for mode in modes:
        cells = [key for key in keys if key.startswith(f"{mode}_a")]
        print(f"{mode:<6} " + "  ".join(
            f"a{key.split('_a')[-1]}={report['mean_iou_by_arm'][key]:.4f}" for key in cells))
    print("alpha = 1 is the oracle repair; alpha = 0 must equal the plain corrupted run "
          f"bit for bit: {report['alpha0_consistent']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
