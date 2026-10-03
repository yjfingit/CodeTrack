#!/usr/bin/env python
"""Audit RGB/TIR frame pairing over the whole LasHeR tree (read-only).

The recovery and gate probes sample scenes through :class:`InfraredIndex`; if that resolver
silently dropped or mispaired frames, their numbers would describe a different frame set
than the one claimed.  This tool runs the resolver over every sequence of both subsets and
reports how each frame was paired, so the coverage claim is reproducible instead of
asserted.

Usage::

    python tools/pairing_audit.py --root /root/autodl-tmp/lab/dataset/LasHeR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recovery_probe import InfraredIndex, IMAGE_SUFFIXES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/root/autodl-tmp/lab/dataset/LasHeR")
    parser.add_argument("--out", default=None, help="optional JSON report path")
    args = parser.parse_args()

    root = Path(args.root)
    report: Dict[str, object] = {"root": str(root), "subsets": {}}
    for subset, list_name in (("testingset", "testingsetList.txt"),
                              ("trainingset", "trainingsetList.txt")):
        sequences = [line.strip() for line in (root / list_name).read_text().splitlines()
                     if line.strip()]
        counts = {"name": 0, "extension": 0, "prefix": 0, "number": 0, "position": 0,
                  "unpaired": 0}
        strategies: Dict[str, int] = {}
        old_rule_unpaired = 0
        order_disagreements = 0
        frames_total = 0
        incomplete: List[Dict[str, object]] = []
        for sequence in sequences:
            visible_dir = root / subset / sequence / "visible"
            infrared_dir = root / subset / sequence / "infrared"
            frames = ([path for path in sorted(visible_dir.iterdir())
                       if path.suffix.lower() in IMAGE_SUFFIXES]
                      if visible_dir.is_dir() else [])
            if not frames:
                incomplete.append({"sequence": sequence, "reason": "no visible frames"})
                continue
            index = InfraredIndex(frames, infrared_dir)
            strategies[index.strategy] = strategies.get(index.strategy, 0) + 1
            missing = 0
            for position, frame in enumerate(frames):
                paired = index.find(frame)
                if paired is None:
                    missing += 1
                elif index.positional is not None and paired != index.positional[position]:
                    # Name-level pairing and sorted order disagree: the two readings of the
                    # directory cannot both be right, so it must be visible in the report.
                    order_disagreements += 1
                # the previous rule, kept only as a baseline for the report
                if not (infrared_dir / ("i" + frame.name[1:])).exists():
                    old_rule_unpaired += 1
            frames_total += len(frames)
            for key, value in index.match_counts.items():
                counts[key] += value
            if missing:
                incomplete.append({"sequence": sequence, "unpaired": missing,
                                   "frames": len(frames), "strategy": index.strategy,
                                   "notes": index.notes})
        entry = {
            "sequences": len(sequences),
            "frames": frames_total,
            "pairing": counts,
            "strategies": strategies,
            "paired_fraction": (frames_total - counts["unpaired"]) / max(frames_total, 1),
            "name_vs_position_disagreements": order_disagreements,
            "old_rule_unpaired_frames": old_rule_unpaired,
            "sequences_with_unpaired_frames": incomplete,
        }
        report["subsets"][subset] = entry
        print(f"{subset}: {len(sequences)} sequences, {frames_total} frames, "
              f"paired {frames_total - counts['unpaired']} "
              f"({100.0 * entry['paired_fraction']:.3f}%)")
        print(f"   by rule: {counts}")
        print(f"   strategy: {strategies}, name-vs-position disagreements: "
              f"{order_disagreements}")
        print(f"   previous \"i\"+name[1:] rule left {old_rule_unpaired} frames unpaired")
        for row in incomplete[:10]:
            print(f"   unpaired: {row}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
