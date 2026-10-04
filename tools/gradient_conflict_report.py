#!/usr/bin/env python
"""Gradient-conflict series per arm, read from the training logs.

Every arm logs `[grad] cos(L_track, L_correct) = ... | |g_track| ... | |g_correct| ...` every
`train.grad_diag_every` steps.  A negative cosine means the tracking and correction objectives are
pushing the shared weights in opposite directions, which is the failure mode the correction branch
has to avoid; a near-zero one means the correction branch is learning something the tracking loss
does not care about.  This is a *secondary* diagnostic (the pre-registered P2 endpoints live in
`docs/results.md` 6.10) but it costs nothing and it separates "the arm tracks worse" from "the two
losses fight in this arm".

Usage::

    python tools/gradient_conflict_report.py --arm A=outputs/p2_A.log --arm D=outputs/p2_D.log
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

PATTERN = re.compile(
    r"\[grad\] cos\(L_track, L_correct\) = (?P<cos>[+-]?\d+\.\d+) \| \|g_track\| "
    r"(?P<g_track>\d+\.\d+) \| \|g_correct\| (?P<g_correct>\d+\.\d+)")


def series(path: Path) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        match = PATTERN.search(line)
        if match:
            out.append({key: float(value) for key, value in match.groupdict().items()})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, metavar="LABEL=LOG")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report: Dict[str, Dict[str, float]] = {}
    print(f"{'arm':<6} {'n':>4} {'mean cos':>9} {'min cos':>8} {'max cos':>8} "
          f"{'mean |g_track|':>14} {'mean |g_correct|':>16} {'ratio':>7}")
    for spec in args.arm:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--arm expects LABEL=LOG, got {spec!r}")
        rows = series(Path(path))
        if not rows:
            print(f"{label:<6} {'-':>4} {'no [grad] lines (was train.grad_diag_every set?)':>60}")
            continue
        cos = [row["cos"] for row in rows]
        gt = [row["g_track"] for row in rows]
        gc = [row["g_correct"] for row in rows]
        mean_cos = sum(cos) / len(cos)
        mean_gt = sum(gt) / len(gt)
        mean_gc = sum(gc) / len(gc)
        report[label] = {"n": len(rows), "mean_cos": mean_cos, "min_cos": min(cos),
                         "max_cos": max(cos), "mean_g_track": mean_gt,
                         "mean_g_correct": mean_gc,
                         "gradient_ratio": (mean_gc / mean_gt) if mean_gt else float("nan")}
        print(f"{label:<6} {len(rows):>4} {mean_cos:>9.3f} {min(cos):>8.3f} {max(cos):>8.3f} "
              f"{mean_gt:>14.3f} {mean_gc:>16.3f} {report[label]['gradient_ratio']:>7.3f}")
    print("\ncos < 0 means the tracking and correction losses pull the shared weights apart; the "
          "gradient ratio says how loud the correction term is relative to tracking.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
