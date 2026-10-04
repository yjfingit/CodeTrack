#!/usr/bin/env python
"""Two inference-protocol differences from the official OSTrack tracker, measured, not assumed.

The reviewer's point was that a low clean SR can be an inference-protocol artefact rather than a
representation limit, and that this had never been checked.  Two concrete differences exist in
this repository, both now behind config flags that default to the historical behaviour:

1. **The crop geometry used to map the box back.**  `_crop_square` clamps the side to
   `MAX_CROP_SIDE_FACTOR` times the longer frame edge and pulls the centre onto the frame, then
   pads.  `Trainer.infer_sequence` mapped the head's normalised box back with the *requested*
   centre and side, so whenever either clamp fired the box was placed wrongly by exactly the
   clamped-minus-requested offset -- in the diverged regime the clamp exists for.  Official
   `map_box_back` uses the crop actually taken.  `eval.crop_mapping="actual"` fixes it; the default
   `"assumed"` keeps the old mapping so archived numbers stay reproducible.
2. **No score-map window.**  The centre head takes a global argmax, so a spurious response far from
   the previous position wins outright.  OSTrack multiplies the score map by a Hann window with a
   `window_influence` before the argmax.  `eval.score_window="hann"` adds it; the default is off.

This tool reports both: how often the clamps fire (so the size of (1) is known independently of any
tracking number), and the paired per-sequence IoU/SR under each protocol setting on the same
sequences and conditions.

Usage::

    python tools/inference_protocol_audit.py --checkpoint outputs/ab_full/final.pth \
        --sequence-list outputs/validation_split_v1/sequences.txt --sequences 20 --frames 200 \
        --out outputs/inference_protocol_audit.json
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

from codetrack.data.transforms.sample import MAX_CROP_SIDE_FACTOR, crop_geometry  # noqa: E402
from codetrack.engine.trainer import Trainer  # noqa: E402
from codetrack.metrics import _box_iou  # noqa: E402
from codetrack.utils.checkpoint import load_checkpoint  # noqa: E402
from codetrack.utils.config import load_config  # noqa: E402
from recovery_probe import list_sequences  # noqa: E402


def clamp_rate(trainer: Trainer, crops: np.ndarray, height: int, width: int
               ) -> Dict[str, float]:
    """How often the crop geometry differs from what the loop assumed when it mapped the box out.

    This is the size of the protocol difference *before* any tracking measurement: if the clamps
    never fire, `crop_mapping` cannot change anything and the flag is a no-op.
    """
    if crops is None or not len(crops):
        return {"frames": 0, "side_clamped": 0.0, "centre_clamped": 0.0, "any_clamped": 0.0}
    side_clamped = centre_clamped = any_clamped = 0
    for crop in crops:
        cx, cy = float(crop[0] + crop[2] / 2), float(crop[1] + crop[3] / 2)
        side = float(np.sqrt(max(crop[2], 1.0) * max(crop[3], 1.0)))
        # the caller multiplies by the search factor before cropping
        search_factor = float(trainer.model.cfg.get("model", {}).get("search_factor", 4.0))
        requested = side * search_factor
        used_cx, used_cy, used_side = crop_geometry(cx, cy, requested, height, width)
        hit_side = abs(used_side - requested) > 1e-6
        hit_centre = abs(used_cx - cx) > 1e-6 or abs(used_cy - cy) > 1e-6
        side_clamped += int(hit_side)
        centre_clamped += int(hit_centre)
        any_clamped += int(hit_side or hit_centre)
    total = len(crops)
    return {"frames": int(total), "side_clamped": side_clamped / total,
            "centre_clamped": centre_clamped / total, "any_clamped": any_clamped / total}


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
    parser.add_argument("--conditions", default="clean,tok_block_rgb_04",
                        help="comma separated: 'clean', 'tok_block_rgb_04', 'rgb_occl_04', ...")
    parser.add_argument("--window-influence", type=float, default=0.5)
    parser.add_argument("--out", default="outputs/inference_protocol_audit.json")
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

    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    protocols = [("historical", "assumed", "none"),
                 ("actual_mapping", "actual", "none"),
                 ("hann_window", "assumed", "hann"),
                 ("actual_and_hann", "actual", "hann")]

    import cv2

    rows: List[Dict[str, object]] = []
    for sequence in sequences:
        # same path convention as Trainer._sequence_frames: <root>/<subset>/<seq>/visible/*.jpg
        frames = sorted((root / args.subset / sequence / "visible").glob("*.jpg"))
        if not frames:
            trainer.logger.warning("skip %s: no frames under %s", sequence,
                                   root / args.subset / sequence / "visible")
            continue
        frame = cv2.imread(str(frames[0]), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            trainer.logger.warning("skip %s: unreadable frame", sequence)
            continue
        height, width = frame.shape[:2]
        entry: Dict[str, object] = {"sequence": sequence, "protocols": {}}
        for condition in conditions:
            corruption = (None if condition == "clean"
                          else {"enabled": True, "token": ["tok_block_erase"]
                                if condition.startswith("tok") else [],
                                "rgb": [], "tir": [], "cross_modal": False,
                                "ratio": 0.4 if condition.endswith("04") else 0.2,
                                "severity": 0.4, "target": "both"})
            if condition == "rgb_occl_04":
                corruption.update({"rgb": ["rgb_occl"], "token": []})
            for label, mapping, window in protocols:
                trainer.crop_mapping = mapping
                trainer.score_window = window
                trainer.window_influence = args.window_influence
                if hasattr(trainer.model.head, "score_window"):
                    trainer.model.head.score_window = window
                    trainer.model.head.window_influence = args.window_influence
                torch.manual_seed(0)
                run = trainer.infer_sequence(root, args.subset, sequence,
                                             max_frames=args.frames, corruption=corruption)
                if len(run["pred"]) == 0:
                    continue
                iou = _box_iou(run["pred"], run["gt"])
                entry["protocols"].setdefault(condition, {})[label] = {
                    "iou_mean": float(iou.mean()),
                    "sr": float((iou > 0.5).mean()),
                    "box_clamps": int(run.get("box_clamps", 0)),
                    "crop_clamps": clamp_rate(trainer, run.get("crops"), height, width),
                }
        rows.append(entry)
        trainer.logger.info("%s done", sequence)

    # restore the historical defaults so the object is left as it was found
    trainer.crop_mapping = "assumed"
    trainer.score_window = "none"
    if hasattr(trainer.model.head, "score_window"):
        trainer.model.head.score_window = "none"

    report: Dict[str, object] = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "overrides": args.override,
        "sequence_list": args.sequence_list,
        "frames": args.frames,
        "max_crop_side_factor": MAX_CROP_SIDE_FACTOR,
        "window_influence": args.window_influence,
        "note": ("'historical' is the shipped protocol (requested crop geometry, no score window); "
                 "the other rows are paired on the same sequences and conditions."),
        "rows": rows,
    }
    for condition in conditions:
        for label, _, _ in protocols:
            values = [row["protocols"][condition][label] for row in rows
                      if condition in row["protocols"] and label in row["protocols"][condition]]
            if not values:
                continue
            report[f"{condition}__{label}"] = {
                "n_sequences": len(values),
                "iou_mean": float(np.mean([v["iou_mean"] for v in values])),
                "sr_mean": float(np.mean([v["sr"] for v in values])),
                "box_clamps": int(sum(v["box_clamps"] for v in values)),
                "crop_clamped_fraction": float(np.mean(
                    [v["crop_clamps"]["any_clamped"] for v in values])),
                "side_clamped_fraction": float(np.mean(
                    [v["crop_clamps"]["side_clamped"] for v in values])),
                "centre_clamped_fraction": float(np.mean(
                    [v["crop_clamps"]["centre_clamped"] for v in values])),
            }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    for condition in conditions:
        base = report.get(f"{condition}__historical")
        if not base:
            continue
        print(f"\n=== {condition} ({base['n_sequences']} sequences) ===")
        print(f"  crop geometry actually clamped on {base['crop_clamped_fraction'] * 100:.2f}% of "
              f"frames (side {base['side_clamped_fraction'] * 100:.2f}%, "
              f"centre {base['centre_clamped_fraction'] * 100:.2f}%)")
        for label, _, _ in protocols:
            row = report.get(f"{condition}__{label}")
            if not row:
                continue
            print(f"  {label:<16} IoU {row['iou_mean']:.4f}  SR {row['sr_mean']:.4f}  "
                  f"clamps {row['box_clamps']:>4}   "
                  f"(dIoU {row['iou_mean'] - base['iou_mean']:+.4f}, "
                  f"dSR {row['sr_mean'] - base['sr_mean']:+.4f})")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
