#!/usr/bin/env python
"""Compare the pre-fix and clamped runs of the same conditions (the clamp audit).

The crop bound and box clamp (``docs/results.md`` 6.15) were added mid-review, so every run
recorded before them may contain frames whose crop was unbounded -- the very frames on which the
tracker had diverged.  This tool prints the two side by side for the conditions that were
re-run, together with the clamp counts of the new run, so the size of the effect on already
published numbers is a measurement rather than a worry.

Usage::

    python tools/clamp_audit_report.py --conditions tok_block_rgb_04,tok_block_tir_04,tok_block_both_02
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

DEFAULT_CONDITIONS = ("tok_block_rgb_04", "tok_block_tir_04", "tok_block_both_02")


def read(directory: Path, condition: str) -> dict:
    path = directory / condition / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", default="outputs/validation_v1/ab_full")
    parser.add_argument("--after", default="outputs/validation_v1/ab_full_clamped")
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    args = parser.parse_args()

    conditions: List[str] = [name.strip() for name in args.conditions.split(",") if name.strip()]
    print(f"{'condition':<22} {'run':<9} {'SR':>6} {'PR':>6} {'NPR':>6} {'iou':>7} "
          f"{'clamped':>8} {'frames':>7}")
    for condition in conditions:
        for label, directory in (("pre-fix", Path(args.before)), ("clamped", Path(args.after))):
            metrics = read(directory, condition)
            if not metrics:
                print(f"{condition:<22} {label:<9} {'-':>6} {'-':>6} {'-':>6} {'-':>7} "
                      f"{'missing':>8}")
                continue
            iou = metrics.get("iou")
            print(f"{condition:<22} {label:<9} {metrics.get('sr', float('nan')) * 100:>6.2f} "
                  f"{metrics.get('pr', float('nan')) * 100:>6.2f} "
                  f"{metrics.get('npr', float('nan')) * 100:>6.2f} "
                  f"{(iou if iou is not None else float('nan')):>7.4f} "
                  f"{str(metrics.get('n_box_clamped_frames', 'n/a')):>8} "
                  f"{str(metrics.get('n_frames', 'n/a')):>7}")
    print("\nA pre-fix row without a counter may contain unclamped diverged crops; the clamped row "
          "shows how many frames were pulled back, and 'iou' is only available in the new run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
