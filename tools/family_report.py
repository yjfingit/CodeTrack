#!/usr/bin/env python
"""Held-out corruption family table: does a detection head transfer across mechanisms?

Section 6.7.1 of ``docs/results.md`` measures the mask-trained head on three zeroing families and
one non-zeroing family, and P4 asks whether a deviation-trained head does better on the
non-zeroing one.  This tool puts those runs side by side, so the answer is a committed command
rather than an ad-hoc comparison:

* `reliability_auroc_rgb` -- AUROC of `1 - r` against the token mask (the head the decoder
  multiplies);
* `syndrome_auroc` -- the parity/syndrome head, which is at or below chance for the mask arm;
* `recovery_gain` / `e_before_mean` / `e_after_mean` -- whether the branch repaired anything.

Usage::

    python tools/family_report.py --arm mask=outputs/validation_v1/family_ \
        --arm deviation=outputs/validation_v1/p4_deviation_family_ \
        --families burst:tok_burst_erase,noise:tok_feat_noise \
        --primary noise --out outputs/family_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

FIELDS = ("reliability_auroc_rgb", "reliability_auroc_tir", "syndrome_auroc",
          "locator_precision_at_5", "recovery_gain", "e_before_mean", "e_after_mean")


def read(prefix: str, family: str) -> Dict[str, float]:
    path = Path(prefix + family) / "metrics.json"
    if not path.exists():
        return {}
    metrics = json.loads(path.read_text())
    return {field: metrics.get(field) for field in FIELDS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, metavar="LABEL=PREFIX",
                        help="prefix such that PREFIX<family>/metrics.json exists; repeatable")
    parser.add_argument("--families", required=True,
                        help="comma separated family:mechanism, where mechanism is free text")
    parser.add_argument("--primary", default=None,
                        help="family whose reliability AUROC is the pre-registered endpoint")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    arms: List[tuple] = []
    for spec in args.arm:
        label, _, prefix = spec.partition("=")
        if not prefix:
            raise SystemExit(f"--arm expects LABEL=PREFIX, got {spec!r}")
        arms.append((label, prefix))
    families: List[tuple] = []
    for spec in args.families.split(","):
        if not spec.strip():
            continue
        name, _, mechanism = spec.partition(":")
        families.append((name.strip(), mechanism.strip() or name.strip()))

    report: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for label, prefix in arms:
        report[label] = {name: read(prefix, name) for name, _ in families}

    print(f"{'arm':<12} {'family':<12} {'mechanism':<22} {'rel_rgb':>8} {'rel_tir':>8} "
          f"{'syn':>7} {'loc@p5':>7} {'rec_gain':>9} {'e_before':>9} {'e_after':>8}")
    for label, _ in arms:
        for name, mechanism in families:
            row = report[label][name]
            if not row:
                print(f"{label:<12} {name:<12} {mechanism:<22} {'missing':>8}")
                continue
            def fmt(key: str) -> str:
                value = row.get(key)
                return "n/a" if value is None else f"{value:.4f}"
            print(f"{label:<12} {name:<12} {mechanism:<22} {fmt('reliability_auroc_rgb'):>8} "
                  f"{fmt('reliability_auroc_tir'):>8} {fmt('syndrome_auroc'):>7} "
                  f"{fmt('locator_precision_at_5'):>7} {fmt('recovery_gain'):>9} "
                  f"{fmt('e_before_mean'):>9} {fmt('e_after_mean'):>8}")

    if args.primary and len(arms) > 1:
        print(f"\nprimary endpoint: reliability_auroc_rgb on the held-out "
              f"{args.primary!r} family")
        baseline_label, baseline_prefix = arms[0]
        baseline = report[baseline_label][args.primary].get("reliability_auroc_rgb")
        for label, _ in arms[1:]:
            value = report[label][args.primary].get("reliability_auroc_rgb")
            if value is None or baseline is None:
                print(f"  {label:<12} missing")
                continue
            print(f"  {label:<12} {value:.4f}  vs {baseline_label} {baseline:.4f}  "
                  f"delta {value - baseline:+.4f}")
            report.setdefault("primary_endpoint", {})[label] = {
                "value": value, "baseline_label": baseline_label, "baseline": baseline,
                "delta": value - baseline, "family": args.primary}

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
