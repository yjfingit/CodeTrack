#!/usr/bin/env python
"""When the tracker loses the target, is the target still inside the crop it is looking at?

Section 6.17.1 measures that most sequences lose the target early and do not recover.  Two very
different stories produce that number, and they imply opposite next steps:

* the target **left the search window** -- then more search range, a global re-detection stage, or
  a motion model is the lever, because the tracker is being asked to find something its input no
  longer contains;
* the target is **still inside the window and the tracker still fails** -- then enlarging the
  search area only adds distractors, and the lever is the representation, the head, or the
  quality/existence estimate that decides when to trust the prediction.

This tool separates them with two measurements that need no training:

1. **Crop containment.**  `Trainer.infer_sequence` now returns the box that drove each frame's
   crop, so for every frame the fraction of the ground-truth box that lies inside the crop square
   is computable directly.  Reported at the failure frame, after the failure, and over all frames.
2. **Oracle re-centring.**  Re-running the same condition with `fixed_boxes = ground truth` crops
   every frame around where the target actually was one frame earlier -- i.e. it removes *only*
   the closed-loop crop error, with the model, the corruption and the head unchanged.  The gap
   between this arm and the free run is an upper bound on what perfect crop placement could buy.

Decision rule, stated before the run:

* containment at failure high (median >= 0.5) and oracle re-centring barely helps -> the search
  window is not the problem; the recovery module is not the next step.
* containment low and oracle re-centring recovers most of the loss -> build the re-detection /
  motion path.
* mixed (some sequences each) -> report the split and pick the majority failure mode.

Usage::

    python tools/search_containment.py --checkpoint outputs/ab_full/final.pth \
        --sequence-list outputs/validation_split_v1/sequences.txt --sequences 60 --frames 200 \
        --out outputs/search_containment_ab_full.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

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


def crop_square(box: np.ndarray, search_factor: float) -> np.ndarray:
    """The square region `Trainer.infer_sequence` actually feeds the model for this crop box.

    The loop takes ``side = sqrt(w * h) * search_factor`` centred on the box centre, so a wide box
    yields a square larger than its own area; containment has to be measured against that square,
    not against the box.
    """
    cx, cy = float(box[0] + box[2] / 2), float(box[1] + box[3] / 2)
    side = float(np.sqrt(max(box[2], 1.0) * max(box[3], 1.0))) * search_factor
    half = side / 2.0
    return np.array([cx - half, cy - half, side, side], dtype=np.float32)


def box_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection area of two xywh boxes."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    return float(width * height)


def containment(gt: np.ndarray, box: np.ndarray, search_factor: float) -> float:
    """Fraction of the ground-truth box inside the crop square implied by ``box``."""
    area = float(max(gt[2], 1e-6) * max(gt[3], 1e-6))
    return box_overlap(gt, crop_square(box, search_factor)) / area


def scale_ratio(gt: np.ndarray, box: np.ndarray, search_factor: float) -> float:
    """Crop side over target side -- how big the window is compared with the object.

    A *correct* prediction gives exactly ``search_factor`` (4.0 here) by construction.  A value
    far above it means the tracker's box has collapsed or exploded, so the crop is mostly
    background and the target is a speck inside a window that still "contains" it.  Containment
    and centre offset cannot see that failure, which is why this third measure is needed.
    """
    gt_side = float(np.sqrt(max(gt[2], 1e-6) * max(gt[3], 1e-6)))
    square = crop_square(box, search_factor)
    return float(max(square[2], 1e-6) / gt_side)


def centre_offset(gt: np.ndarray, box: np.ndarray, search_factor: float) -> float:
    """Distance between the target centre and the crop centre, in units of the crop side.

    Containment alone is a weak instrument: the crop square is ``search_factor`` times the
    object's linear size, so the target stays *inside* the crop through a drift of more than one
    object width.  What actually determines whether the head can find it is how far off-centre it
    is.  0 is a perfectly framed frame, 0.5 puts the target on the crop boundary.
    """
    square = crop_square(box, search_factor)
    side = float(max(square[2], 1e-6))
    gx, gy = float(gt[0] + gt[2] / 2), float(gt[1] + gt[3] / 2)
    cx, cy = float(square[0] + square[2] / 2), float(square[1] + square[3] / 2)
    return float(np.hypot(gx - cx, gy - cy) / side)


def first_failure(iou: np.ndarray, threshold: float, patience: int):
    """First frame of a run of ``patience`` consecutive frames below ``threshold`` (event start)."""
    run = 0
    for index, value in enumerate(iou):
        run = run + 1 if float(value) < threshold else 0
        if run >= patience:
            return index - patience + 1
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/experiment/lasher_vitb_corrupt.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--root", default="/root/autodl-tmp/lab/dataset/LasHeR")
    parser.add_argument("--subset", default="testingset")
    parser.add_argument("--sequence-list", default=None)
    parser.add_argument("--sequences", type=int, default=60)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--token", default="tok_block_erase")
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--severity", type=float, default=0.4)
    parser.add_argument("--target", default="both", choices=["both", "rgb", "tir"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--containment-threshold", type=float, default=0.5,
                        help="above this the target is considered to be inside the crop")
    parser.add_argument("--framed-offset", type=float, default=0.25,
                        help="centre offset (in crop sides) below which the target counts as "
                             "well framed rather than merely present")
    parser.add_argument("--framed-scale", type=float, nargs=2, default=(2.0, 8.0),
                        metavar=("LOW", "HIGH"),
                        help="window-to-object side ratio range that counts as well framed; a "
                             "correct prediction gives exactly search_factor (4.0)")
    parser.add_argument("--clean", action="store_true",
                        help="also run the same measurement without corruption")
    parser.add_argument("--out", default="outputs/search_containment.json")
    args = parser.parse_args()

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

    search_factor = float(cfg.get("model", {}).get("search_factor", 4.0))
    corruption = {"enabled": True, "token": [args.token], "rgb": [], "tir": [],
                  "cross_modal": False, "ratio": args.ratio, "severity": args.severity,
                  "target": args.target}

    conditions = [("corrupt", corruption)]
    if args.clean:
        conditions.insert(0, ("clean", None))

    rows: List[Dict[str, object]] = []
    for sequence in sequences:
        entry: Dict[str, object] = {"sequence": sequence, "conditions": {}}
        for label, condition in conditions:
            torch.manual_seed(0)
            free = trainer.infer_sequence(root, args.subset, sequence,
                                          max_frames=args.frames, corruption=condition)
            if len(free["pred"]) == 0:
                continue
            pred, gt, crops = free["pred"], free["gt"], free.get("crops")
            iou = _box_iou(pred, gt)
            failure = first_failure(iou, args.threshold, args.patience)
            # Oracle re-centring: the crop schedule becomes the ground truth *lagged by one
            # frame*, which is exactly what a perfect tracker would have used.  Only the crop
            # placement changes; model, corruption and head are untouched.
            torch.manual_seed(0)
            oracle = trainer.infer_sequence(root, args.subset, sequence,
                                            max_frames=args.frames, corruption=condition,
                                            fixed_boxes=gt)
            oracle_iou = _box_iou(oracle["pred"], oracle["gt"])

            contain = (np.array([containment(gt[i], crops[i], search_factor)
                                 for i in range(min(len(gt), len(crops)))])
                       if crops is not None and len(crops) else np.zeros(0))
            offset = (np.array([centre_offset(gt[i], crops[i], search_factor)
                                for i in range(min(len(gt), len(crops)))])
                      if crops is not None and len(crops) else np.zeros(0))
            ratio = (np.array([scale_ratio(gt[i], crops[i], search_factor)
                               for i in range(min(len(gt), len(crops)))])
                     if crops is not None and len(crops) else np.zeros(0))
            record: Dict[str, object] = {
                "frames": int(len(pred)),
                "free_iou_mean": float(iou.mean()),
                "free_sr": float((iou > 0.5).mean()),
                "oracle_crop_iou_mean": float(oracle_iou.mean()),
                "oracle_crop_sr": float((oracle_iou > 0.5).mean()),
                "failure_frame": failure,
                "recovered_after_failure": (
                    None if failure is None else
                    bool(np.any(iou[failure + args.patience:] >= args.threshold))),
            }
            if contain.size:
                record.update({
                    "containment_mean": float(contain.mean()),
                    "containment_median": float(np.median(contain)),
                    "containment_at_failure": (None if failure is None else
                                               float(contain[min(failure, contain.size - 1)])),
                    "containment_after_failure": (None if failure is None else
                                                  float(contain[failure:].mean())),
                    "inside_fraction": float((contain >= args.containment_threshold).mean()),
                    # the framing measure: how far off-centre the target is, and how often it is
                    # off-centre enough that the head has to search the whole window
                    "offset_median": float(np.median(offset)),
                    "offset_after_failure_median": (None if failure is None else
                                                    float(np.median(offset[failure:]))),
                    "scale_ratio_median": float(np.median(ratio)),
                    "scale_ratio_after_failure_median": (None if failure is None else
                                                         float(np.median(ratio[failure:]))),
                    # "well framed" = on-centre AND at the intended window-to-object scale, which
                    # is the conjunction a re-detection decision would have to test
                    "framed_fraction": float(np.mean(
                        (offset <= args.framed_offset)
                        & (ratio >= args.framed_scale[0]) & (ratio <= args.framed_scale[1]))),
                })
                record["box_clamps"] = int(free.get("box_clamps", 0))
            entry["conditions"][label] = record
        rows.append(entry)
        summary = entry["conditions"].get("corrupt") or {}
        trainer.logger.info(
            "%s: free_iou=%.3f oracle_crop_iou=%.3f containment_after_failure=%s failure=%s",
            sequence, summary.get("free_iou_mean", float("nan")),
            summary.get("oracle_crop_iou_mean", float("nan")),
            summary.get("containment_after_failure"), summary.get("failure_frame"))

    report: Dict[str, object] = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "overrides": args.override,
        "sequence_list": args.sequence_list,
        "frames": args.frames,
        "threshold": args.threshold,
        "patience": args.patience,
        "containment_threshold": args.containment_threshold,
        "framed_offset": args.framed_offset,
        "corruption": {"token": args.token, "ratio": args.ratio, "severity": args.severity,
                       "target": args.target},
        "note": ("containment is the fraction of the ground-truth box inside the crop square the "
                 "loop actually used; oracle_crop_* re-runs the same condition with the crop "
                 "schedule replaced by the lagged ground truth, so the gap to free is an upper "
                 "bound on what perfect crop placement buys."),
        "rows": rows,
    }
    for label, _ in conditions:
        records = [row["conditions"][label] for row in rows if label in row["conditions"]]
        if not records:
            continue
        failed = [r for r in records if r["failure_frame"] is not None]
        failed_after = [r["containment_after_failure"] for r in failed
                        if r.get("containment_after_failure") is not None]
        report[f"summary_{label}"] = {
            "n_sequences": len(records),
            "failure_fraction": float(len(failed) / len(records)),
            "free_iou_mean": float(np.mean([r["free_iou_mean"] for r in records])),
            "free_sr_mean": float(np.mean([r["free_sr"] for r in records])),
            "oracle_crop_iou_mean": float(np.mean([r["oracle_crop_iou_mean"] for r in records])),
            "oracle_crop_sr_mean": float(np.mean([r["oracle_crop_sr"] for r in records])),
            "oracle_crop_gain_iou": float(np.mean([r["oracle_crop_iou_mean"] - r["free_iou_mean"]
                                                   for r in records])),
            "containment_mean_overall": float(np.mean(
                [r["containment_mean"] for r in records if "containment_mean" in r]))
            if any("containment_mean" in r for r in records) else None,
            "containment_after_failure_median": float(np.median(failed_after))
            if failed_after else None,
            "offset_median": float(np.mean([r["offset_median"] for r in records
                                            if "offset_median" in r]))
            if any("offset_median" in r for r in records) else None,
            "offset_after_failure_median": float(np.median(
                [r["offset_after_failure_median"] for r in failed
                 if r.get("offset_after_failure_median") is not None]))
            if any(r.get("offset_after_failure_median") is not None for r in failed) else None,
            "scale_ratio_after_failure_median": float(np.median(
                [r["scale_ratio_after_failure_median"] for r in failed
                 if r.get("scale_ratio_after_failure_median") is not None]))
            if any(r.get("scale_ratio_after_failure_median") is not None
                   for r in failed) else None,
            "framed_fraction": float(np.mean([r["framed_fraction"] for r in records
                                              if "framed_fraction" in r]))
            if any("framed_fraction" in r for r in records) else None,
            # The two halves of the decision rule, in one place:
            "sequences_inside_crop_after_failure": int(sum(
                1 for value in failed_after if value >= args.containment_threshold)),
            "sequences_outside_crop_after_failure": int(sum(
                1 for value in failed_after if value < args.containment_threshold)),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    for label, _ in conditions:
        summary = report.get(f"summary_{label}")
        if not summary:
            continue
        print(f"\n=== {label} ({summary['n_sequences']} sequences) ===")
        print(f"  failure within horizon      {summary['failure_fraction'] * 100:.1f}%")
        print(f"  free          IoU {summary['free_iou_mean']:.4f}  "
              f"SR {summary['free_sr_mean']:.4f}")
        print(f"  oracle crop   IoU {summary['oracle_crop_iou_mean']:.4f}  "
              f"SR {summary['oracle_crop_sr_mean']:.4f}  "
              f"(gain {summary['oracle_crop_gain_iou']:+.4f} IoU)")
        if summary["containment_after_failure_median"] is not None:
            print(f"  containment after failure   median "
                  f"{summary['containment_after_failure_median']:.3f}  "
                  f"inside/outside = {summary['sequences_inside_crop_after_failure']}/"
                  f"{summary['sequences_outside_crop_after_failure']}")
            print(f"  centre offset               overall {summary['offset_median']:.3f}  "
                  f"after failure {summary['offset_after_failure_median']:.3f} crop sides")
            print(f"  window/object side ratio    after failure "
                  f"{summary['scale_ratio_after_failure_median']:.2f} "
                  f"(a correct prediction gives {search_factor:g})")
            print(f"  well framed fraction        {summary['framed_fraction']:.3f} "
                  f"(offset <= {args.framed_offset}, ratio in "
                  f"[{args.framed_scale[0]:g}, {args.framed_scale[1]:g}])")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
