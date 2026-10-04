#!/usr/bin/env python
"""Power planning from measured, per-sequence paired data.

The reviewer's point: an MDE table built from assumed variances is decoration.  This tool reads
the per-sequence SR of two evaluation runs and estimates the quantity the claim is actually
tested on -- the paired difference in differences

    D_i = drop_B(i) - drop_A(i),   drop(i) = SR_clean(i) - SR_condition(i)

so sigma_D is measured on the same sequences, with the same corruption, that the final test
will use.  It then reports the minimum detectable effect under the pre-registered testing plan:

    MDE = (z_{1 - alpha/(2m)} + z_{1-beta}) * sigma_D / sqrt(n)

with ``m`` the size of the Holm family (condition comparisons + interactions), ``n`` the number
of sequences, and alpha = 0.05, power = 0.80.  The reverse direction (sequences needed for an
observed effect) is reported too.

Two modes:

* ``--compare A B`` -- difference in differences between two runs (the robustness claim);
* a single ``--run``    -- the plain paired drop against clean (what a per-arm test sees).

Usage::

    python tools/power_plan.py --compare ab_full ab_nodec_corr \
        --run ab_full=outputs/validation_v1/ab_full \
        --run ab_nodec_corr=outputs/validation_v1/ab_nodec_corr \
        --condition tok_block_rgb_02 --reference clean --sequences 60 --comparisons 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats


def per_sequence(path: Path, metric: str = "sr") -> Dict[str, float]:
    data = json.loads(path.read_text())
    key = "iou_mean" if metric == "iou" else metric
    rows = data.get("per_sequence", [])
    missing = [row for row in rows if key not in row]
    if missing:
        raise SystemExit(f"{path}: {len(missing)} sequences lack {key!r}; that run predates the "
                         "mean-IoU endpoint")
    return {row["sequence"]: float(row[key]) for row in rows}


def z(p: float) -> float:
    return float(stats.norm.ppf(p))


def mde(sigma: float, n: int, comparisons: int, alpha: float = 0.05,
        power: float = 0.80) -> float:
    if sigma <= 0 or n < 2:
        return float("nan")
    return (z(1.0 - alpha / (2.0 * max(comparisons, 1))) + z(power)) * sigma / np.sqrt(n)


def required_n(sigma: float, effect: float, comparisons: int, alpha: float = 0.05,
               power: float = 0.80) -> Optional[float]:
    if sigma <= 0 or effect <= 0:
        return None
    factor = (z(1.0 - alpha / (2.0 * max(comparisons, 1))) + z(power)) * sigma / effect
    return float(factor ** 2)


def holm_rejected(p_values: np.ndarray, alpha: float = 0.05) -> set:
    """Indices Holm-Bonferroni rejects, following the real step-down procedure."""
    order = np.argsort(np.asarray(p_values, dtype=float))
    m = len(order)
    rejected: set = set()
    for rank, index in enumerate(order, start=1):
        if float(p_values[index]) <= alpha / (m - rank + 1):
            rejected.add(int(index))
        else:
            break
    return rejected


def simulate_power(differences: np.ndarray, effect: float, comparisons: int, sequences: int,
                   reps: int = 2000, alpha: float = 0.05, seed: int = 0,
                   correlation: str = "shared") -> Dict[str, float]:
    """Monte-Carlo power under the actual Holm procedure and the *measured* variance.

    ``differences`` are the observed per-sequence effects; the simulation resamples them
    (bootstrap-style, mean-centred) so no normality or variance assumption is added, shifts the
    target comparison to ``effect``, leaves the rest of the family at zero, and applies Holm
    exactly as the final analysis will.

    Two corrections to the first version, both of which made the reported numbers optimistic:

    * **FWER means "any rejection in the family", not "the target was rejected".**  The old code
      checked ``0 in holm_rejected(...)``, so the ``effect = 0`` row estimated the *per-hypothesis*
      type-I rate (roughly alpha) rather than the family-wise error rate it claimed to report.
    * **Comparisons share the same sequences.**  The old code drew an independent sample for every
      comparison, which destroys the positive correlation between condition contrasts measured on
      the same sequences -- a family of highly correlated tests is rejected much less often than a
      family of independent ones, so the independent draw made the family look more powerful (and
      the FWER look worse) than it is.  ``correlation="independent"`` keeps the old behaviour
      available for comparison only.
    """
    values = np.asarray(differences, dtype=float)
    if values.ndim == 1:
        # One measured contrast.  The other members of the family are then modelled as *perfectly
        # correlated* with it (same resample, no shift), which is the conservative end of the
        # range: Holm's threshold applies to the family's smallest p-value, so perfectly
        # dependent null tests are rejected least often.  Passing an (m x n) array instead lets
        # each comparison carry its own measured per-sequence effect, which is the realistic
        # middle ground and what a real family looks like.
        values = np.tile(values[None, :], (max(1, comparisons), 1))
    if values.ndim != 2 or values.shape[1] < 2:
        return {"power": float("nan"), "n_sequences": sequences, "comparisons": comparisons,
                "effect_points": float(effect), "reps": reps}
    if values.shape[0] < comparisons:
        values = np.tile(values[:1], (comparisons, 1))
    comparisons = min(comparisons, values.shape[0])
    if correlation not in ("shared", "independent"):
        raise ValueError(f"unknown correlation mode {correlation!r}")
    centred = values[:comparisons] - values[:comparisons].mean(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    target_rejections = 0
    any_rejections = 0
    for _ in range(reps):
        shared = rng.integers(0, centred.shape[1], size=sequences)
        p_values = np.empty(comparisons, dtype=float)
        for index in range(comparisons):
            shift = float(effect) if index == 0 else 0.0
            draw = (shared if correlation == "shared"
                    else rng.integers(0, centred.shape[1], size=sequences))
            sample = centred[index][draw] + shift
            if np.allclose(sample, sample[0]):
                p_values[index] = 1.0
            else:
                p_values[index] = float(stats.ttest_1samp(sample, 0.0).pvalue)
        rejected = holm_rejected(p_values, alpha)
        if rejected:
            any_rejections += 1
        if 0 in rejected:
            target_rejections += 1
    return {"power": target_rejections / reps, "n_sequences": sequences,
            "comparisons": comparisons, "effect_points": float(effect), "reps": reps,
            "alpha": alpha, "correlation": correlation,
            # Under effect = 0 this is the family-wise error rate: P(at least one rejection).
            "family_wise_rejection_rate": any_rejections / reps}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], help="label=directory, repeatable")
    parser.add_argument("--compare", nargs=2, default=None, metavar=("LABEL_A", "LABEL_B"),
                        help="difference in differences between two runs")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--reference", default="clean")
    parser.add_argument("--metric", default="sr", choices=["sr", "pr", "npr", "iou"],
                        help="per-sequence metric the paired difference is computed on; the "
                             "pre-registered primary is SR.  'iou' reads the additional "
                             "mean-IoU endpoint and is not thresholded, so it is the lower-"
                             "variance option for power planning")
    parser.add_argument("--sequences", type=int, default=60,
                        help="planned number of sequences for the final test")
    parser.add_argument("--comparisons", type=int, default=1,
                        help="size of the Holm family (condition comparisons + interactions)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--simulate", action="store_true",
                        help="Monte-Carlo power under the real Holm procedure, resampling the "
                             "measured per-sequence differences instead of assuming a variance")
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0, help="simulation seed")
    parser.add_argument("--effects", default="0,2,3,5,8,12",
                        help="effect sizes in the metric's points for the power curve; 0 gives "
                             "the family-wise error rate")
    parser.add_argument("--families", default="1,4,8",
                        help="Holm family sizes to simulate (condition comparisons + interactions)")
    parser.add_argument("--correlation", default="shared", choices=["shared", "independent"],
                        help="'shared' resamples one sequence set per repetition and reuses it for "
                             "every comparison in the family, which is how the real analysis "
                             "behaves (all contrasts are measured on the same sequences); "
                             "'independent' reproduces the old, overly optimistic draw")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    runs: Dict[str, Path] = {}
    for spec in args.run:
        label, _, directory = spec.partition("=")
        if not directory:
            raise SystemExit(f"--run expects label=directory, got {spec!r}")
        runs[label] = Path(directory)
    if args.compare:
        for label in args.compare:
            if label not in runs:
                raise SystemExit(f"--compare needs --run {label}=<dir>")

    report: Dict[str, object] = {
        "condition": args.condition, "reference": args.reference, "metric": args.metric,
        "planned_sequences": args.sequences, "comparisons": args.comparisons,
        "alpha": args.alpha, "power": args.power, "runs": {},
    }

    for label, directory in runs.items():
        clean = per_sequence(directory / args.reference / "metrics.json", args.metric)
        condition = per_sequence(directory / args.condition / "metrics.json", args.metric)
        shared = sorted(set(clean) & set(condition))
        drop = np.array([clean[s] - condition[s] for s in shared]) * 100.0   # SR points
        entry = {
            "n_sequences": len(shared),
            "mean_drop_points": float(drop.mean()) if drop.size else float("nan"),
            "sigma_drop_points": float(drop.std(ddof=1)) if drop.size > 1 else float("nan"),
            "mde_points_at_planned_n": mde(float(drop.std(ddof=1)) if drop.size > 1 else 0.0,
                                           args.sequences, args.comparisons,
                                           args.alpha, args.power),
        }
        report["runs"][label] = entry
        print(f"{label:<16} n={len(shared)} mean drop {entry['mean_drop_points']:+7.2f} pts | "
              f"sigma_drop {entry['sigma_drop_points']:6.2f} pts | MDE(n={args.sequences}, "
              f"m={args.comparisons}) {entry['mde_points_at_planned_n']:.2f} pts")

    if args.compare:
        label_a, label_b = args.compare
        clean_a = per_sequence(runs[label_a] / args.reference / "metrics.json", args.metric)
        cond_a = per_sequence(runs[label_a] / args.condition / "metrics.json", args.metric)
        clean_b = per_sequence(runs[label_b] / args.reference / "metrics.json", args.metric)
        cond_b = per_sequence(runs[label_b] / args.condition / "metrics.json", args.metric)
        shared = sorted(set(clean_a) & set(cond_a) & set(clean_b) & set(cond_b))
        if len(shared) < 2:
            raise SystemExit("not enough shared sequences for the comparison")
        drop_a = np.array([clean_a[s] - cond_a[s] for s in shared]) * 100.0
        drop_b = np.array([clean_b[s] - cond_b[s] for s in shared]) * 100.0
        difference = drop_b - drop_a                     # D > 0: A degrades less
        sigma = float(difference.std(ddof=1))
        observed = float(difference.mean())
        entry = {
            "n_sequences": len(shared),
            "mean_difference_points": observed,
            "sigma_difference_points": sigma,
            "mde_points_at_planned_n": mde(sigma, args.sequences, args.comparisons,
                                           args.alpha, args.power),
            "sequences_for_observed_effect": required_n(sigma, abs(observed),
                                                        args.comparisons, args.alpha,
                                                        args.power),
            "sign_convention": "positive = arm A degrades less",
        }
        report["difference_in_differences"] = entry
        print(f"difference in differences {label_b} - {label_a}: observed {observed:+.2f} pts | "
              f"sigma_D {sigma:.2f} pts | MDE(n={args.sequences}, m={args.comparisons}) "
              f"{entry['mde_points_at_planned_n']:.2f} pts | sequences needed for the observed "
              f"effect: {entry['sequences_for_observed_effect']}")
        print("MDE uses the normal approximation with a Bonferroni-style family correction; "
              "it is a design estimate from measured paired variance, not a guarantee.")

    if args.simulate:
        # the quantity being tested: the paired difference in differences where available,
        # otherwise the single-arm paired drop
        if "difference_in_differences" in report:
            clean_a = per_sequence(runs[args.compare[0]] / args.reference / "metrics.json",
                                   args.metric)
            cond_a = per_sequence(runs[args.compare[0]] / args.condition / "metrics.json",
                                  args.metric)
            clean_b = per_sequence(runs[args.compare[1]] / args.reference / "metrics.json",
                                   args.metric)
            cond_b = per_sequence(runs[args.compare[1]] / args.condition / "metrics.json",
                                  args.metric)
            shared = sorted(set(clean_a) & set(cond_a) & set(clean_b) & set(cond_b))
            observed = np.array([(clean_b[s] - cond_b[s]) - (clean_a[s] - cond_a[s])
                                 for s in shared]) * 100.0
            unit = "difference in differences (points)"
        else:
            label = next(iter(runs))
            clean = per_sequence(runs[label] / args.reference / "metrics.json", args.metric)
            condition = per_sequence(runs[label] / args.condition / "metrics.json", args.metric)
            shared = sorted(set(clean) & set(condition))
            observed = np.array([clean[s] - condition[s] for s in shared]) * 100.0
            unit = f"paired drop of {label} (points)"
        effects = [float(value) for value in args.effects.split(",") if value.strip()]
        families = [int(value) for value in args.families.split(",") if value.strip()]
        grid = []
        print(f"\nMonte-Carlo power on {len(shared)} measured per-sequence differences "
              f"({unit}); {args.reps} repetitions per cell, the real Holm step-down, "
              f"correlation={args.correlation}")
        header = "  ".join(f"m={m:<3}" for m in families)
        print(f"{'effect':>8}  {header}")
        for effect in effects:
            cells = []
            for size in families:
                result = simulate_power(observed, effect, size, args.sequences, args.reps,
                                        args.alpha, args.seed, args.correlation)
                grid.append(result)
                cells.append(f"{result['power']:>5.3f}")
            label = "FWER" if effect == 0 else f"{effect:g}"
            print(f"{label:>8}  " + "  ".join(cells))
            if effect == 0:
                # The row labelled FWER must report P(any rejection), not P(this one rejection);
                # the two differ by roughly the family size (docs/results.md 6.13).
                rates = "  ".join(f"{r['family_wise_rejection_rate']:>5.3f}"
                                  for r in grid[-len(families):])
                print(f"{'P(any)':>8}  {rates}")
        report["simulation"] = {"unit": unit, "grid": grid, "observed": observed.tolist(),
                                "correlation": args.correlation}

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
