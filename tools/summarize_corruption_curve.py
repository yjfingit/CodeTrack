#!/usr/bin/env python
"""Summarize paired sequence-level clean-to-corrupt SR changes with bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


def read_by_sequence(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text())
    if "per_sequence" not in data:
        raise ValueError(f"{path} has no per_sequence metrics; rerun the evaluation")
    return {row["sequence"]: float(row["sr"]) for row in data["per_sequence"]}


def bootstrap_mean_interval(values: np.ndarray, rng: np.random.Generator,
                            draws: int) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--token", default="tok_block_erase")
    parser.add_argument("--target", default="rgb")
    parser.add_argument("--ratios", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.out_dir)
    clean = read_by_sequence(output / "clean" / "metrics.json")
    rng = np.random.default_rng(args.seed)
    report = []
    print("ratio  n  clean_SR  corrupt_SR  relative_drop_%  paired_95%_CI")
    for text_ratio in args.ratios.split(","):
        ratio = float(text_ratio)
        ratio_name = f"{ratio:.1f}"
        path = output / f"{args.token}_{args.target}_{ratio_name}" / "metrics.json"
        corrupt = read_by_sequence(path)
        names = sorted(clean.keys() & corrupt.keys())
        if not names:
            raise ValueError(f"no paired sequences between clean and {path}")
        clean_values = np.array([clean[name] for name in names], dtype=np.float64)
        corrupt_values = np.array([corrupt[name] for name in names], dtype=np.float64)
        relative_drop = 100.0 * (clean_values - corrupt_values) / np.maximum(clean_values, 1e-8)
        low, high = bootstrap_mean_interval(relative_drop, rng, args.draws)
        row = {
            "ratio": ratio,
            "n_sequences": len(names),
            "clean_sr_mean": float(clean_values.mean()),
            "corrupt_sr_mean": float(corrupt_values.mean()),
            "relative_drop_percent_mean": float(relative_drop.mean()),
            "relative_drop_percent_ci95": [low, high],
            "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
        }
        report.append(row)
        print(f"{ratio:>4.1f} {len(names):>3} {row['clean_sr_mean']:>9.4f} "
              f"{row['corrupt_sr_mean']:>11.4f} "
              f"{row['relative_drop_percent_mean']:>15.2f} "
              f"[{low:.2f}, {high:.2f}]")

    (output / "paired_curve_summary.json").write_text(json.dumps(report, indent=2))
    print("Intervals resample sequences, not frames; they describe this selected validation set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
