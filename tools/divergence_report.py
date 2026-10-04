#!/usr/bin/env python
"""Report the closed-loop divergence rate of every evaluation run.

``clamp_box`` (``docs/results.md`` 6.15) bounds a diverged prediction, and the evaluator counts
the frames it had to touch.  A run where the tracker clips its own box is a run whose metrics
describe a *clamped* trajectory, so the rate belongs next to every SR/PR number rather than in a
log line.  This tool puts the counts side by side across runs and conditions.

Coverage note: the counter was added during the review round, so runs recorded before it report
"unknown" here -- that is deliberate, because "0 %" and "we did not measure it" must not look the
same (a run predating the counter can still contain unclamped diverged frames).

Usage::

    python tools/divergence_report.py --run ab_full_clamped=outputs/validation_v1/ab_full_clamped \
        --run mlp=outputs/validation_v1/ab_mlp_corr --out outputs/divergence_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def rates(path: Path) -> List[Dict[str, Optional[float]]]:
    """Per-condition divergence counts of one run directory."""
    rows: List[Dict[str, Optional[float]]] = []
    for condition in sorted(path.iterdir()):
        metrics_path = condition / "metrics.json"
        if not (condition.is_dir() and metrics_path.exists()):
            continue
        metrics = json.loads(metrics_path.read_text())
        frames = metrics.get("n_frames")
        clamped = metrics.get("n_box_clamped_frames")
        if frames is None or clamped is None:
            # reconstruct from the per-sequence rows when only those were recorded
            per_sequence = metrics.get("per_sequence") or []
            if per_sequence and "frames" in per_sequence[0]:
                frames = sum(int(row.get("frames", 0)) for row in per_sequence)
                clamped = sum(int(row.get("box_clamps", 0)) for row in per_sequence)
        if frames is None or clamped is None:
            rows.append({"condition": condition.name, "frames": None, "clamped": None,
                         "divergence_rate": None, "note": "recorded before the counter existed"})
            continue
        rows.append({
            "condition": condition.name,
            "frames": int(frames),
            "clamped": int(clamped),
            "divergence_rate": (float(clamped) / float(frames)) if frames else None,
            # the worst sequence is often the interesting one: a single diverged sequence can
            # dominate a mean IoU or SR number without showing up in the run-level rate
            "worst_sequence": max(
                ((row.get("sequence"), int(row.get("box_clamps", 0)))
                 for row in metrics.get("per_sequence", []) or []),
                key=lambda item: item[1], default=(None, 0))[0],
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIR",
                        help="repeatable")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report: Dict[str, List[Dict[str, Optional[float]]]] = {}
    for spec in args.run:
        label, _, directory = spec.partition("=")
        if not directory:
            raise SystemExit(f"--run expects LABEL=DIR, got {spec!r}")
        rows = rates(Path(directory))
        report[label] = rows
        known = [row for row in rows if row["divergence_rate"] is not None]
        total_frames = sum(row["frames"] for row in known)
        total_clamped = sum(row["clamped"] for row in known)
        overall = (total_clamped / total_frames) if total_frames else float("nan")
        print(f"\n=== {label} ({directory}) ===")
        print(f"{'condition':<28} {'frames':>8} {'clamped':>8} {'rate':>8}  worst sequence")
        for row in rows:
            if row["divergence_rate"] is None:
                print(f"{row['condition']:<28} {'-':>8} {'-':>8} {'unknown':>8}  "
                      f"(counter not recorded)")
                continue
            print(f"{row['condition']:<28} {row['frames']:>8} {row['clamped']:>8} "
                  f"{row['divergence_rate'] * 100:>7.2f}%  {row['worst_sequence']}")
        print(f"{'ALL conditions with counters':<28} {total_frames:>8} {total_clamped:>8} "
              f"{overall * 100:>7.2f}%")
        print(f"{len(rows) - len(known)} of {len(rows)} conditions have no counter "
              f"(recorded before the fix); their frames may include unclamped diverged crops.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
