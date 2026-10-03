#!/usr/bin/env python
"""Paired, per-sequence summary of a condition matrix produced by ``validate_sequences.sh``.

Sequences are the sampling unit, never frames: adjacent frames of one sequence are not
independent, so a frame-level interval would be far too narrow.  Every condition is paired
with the same run's reference (``clean`` by default) sequence by sequence, which is also
what makes the two checkpoints comparable -- they see the same sequences under the same
seed and the same corruption protocol.

Reported per (run, condition):

* ``sr_delta`` / ``relative_drop_percent`` plus a bootstrap interval that resamples
  sequences;
* paired t-test and Wilcoxon signed-rank p-values, with Holm-corrected p across the
  conditions of that run;
* the corpus-level error-correction numbers (syndrome AUROC, locator P@5, recovery gain,
  clean-region damage) exactly as the evaluator reported them, so they are not re-derived
  here.

With ``--compare A B`` the script also reports, per condition, the paired sequence-level SR
difference between two runs (A - B), which is the full-model vs no-decoder ablation.

Usage::

    python tools/summarize_paired_conditions.py \
        --run full=outputs/validation_ab_full_v1 \
        --run nodec=outputs/validation_ab_nodec_v1 \
        --compare full nodec --out outputs/validation_summary_v1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

METRIC_KEYS = ("sr", "pr", "npr")
POOLED_KEYS = ("syndrome_auroc", "syndrome_density_spearman", "locator_precision_at_5",
               "locator_precision_chance", "recovery_gain", "damage_clean",
               "e_after_over_before", "hinge_mean", "gain_active_fraction")
# Pre-registered strength screen (docs/corruption_protocol.md section 6).  An engineering
# screen, not a general law: a corruption level is "calibrated" when it costs the frozen
# no-decoder checkpoint 20-40 % relative SR while leaving it 60-80 % of its clean SR, with a
# paired sequence-level interval that excludes zero.
SCREEN_RELATIVE_DROP = (20.0, 40.0)
SCREEN_RETAINED = (60.0, 80.0)


def read_run(path: Path) -> Dict[str, Dict]:
    """All condition directories of one evaluation run, keyed by directory name."""
    conditions: Dict[str, Dict] = {}
    for child in sorted(path.iterdir()):
        metrics_path = child / "metrics.json"
        if child.is_dir() and metrics_path.exists():
            conditions[child.name] = json.loads(metrics_path.read_text())
    return conditions


def per_sequence(metrics: Dict, metric: str) -> Dict[str, float]:
    return {row["sequence"]: float(row[metric]) for row in metrics.get("per_sequence", [])}


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator, draws: int
                       ) -> Tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def difference_in_differences(drop_a: np.ndarray, drop_b: np.ndarray,
                              rng: np.random.Generator, draws: int) -> Dict[str, float]:
    """Is arm A's corruption-induced drop smaller than arm B's, sequence by sequence?

    Both arms are paired on the same sequences, so this is the robustness claim itself:
    ``mean_difference < 0`` means the first arm loses less SR under the same corruption.
    """
    diff = drop_a - drop_b
    low, high = bootstrap_interval(diff, rng, draws)
    t_p = float("nan")
    w_p = float("nan")
    if len(diff) > 1 and np.any(diff != 0):
        t_p = float(stats.ttest_rel(drop_a, drop_b).pvalue)
        try:
            w_p = float(stats.wilcoxon(drop_a, drop_b).pvalue)
        except ValueError:
            pass
    return {
        "n_sequences": int(len(diff)),
        "mean_drop_a": float(drop_a.mean()),
        "mean_drop_b": float(drop_b.mean()),
        "mean_difference": float(diff.mean()),
        "difference_ci95": [low, high],
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
        "paired_t_p": t_p,
        "wilcoxon_p": w_p,
    }


def holm(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values, order preserved."""
    adjusted = [1.0] * len(p_values)
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    running = 0.0
    for rank, index in enumerate(order):
        scaled = (len(p_values) - rank) * p_values[index]
        running = max(running, min(1.0, scaled))
        adjusted[index] = running
    return adjusted


def paired_stats(reference: np.ndarray, condition: np.ndarray, rng: np.random.Generator,
                 draws: int) -> Dict[str, float]:
    delta = condition - reference
    drop = -delta                                    # positive = corruption hurt
    relative = 100.0 * drop / np.maximum(reference, 1e-8)    # positive = corruption hurt
    low, high = bootstrap_interval(relative, rng, draws)
    drop_low, drop_high = bootstrap_interval(drop, rng, draws)
    out = {
        "mean_reference": float(reference.mean()),
        "mean_condition": float(condition.mean()),
        "mean_delta": float(delta.mean()),
        # Absolute, in SR points: this is the usable effect size here, because ~half of the
        # validation sequences have a clean SR near zero and a per-sequence ratio explodes
        # on those (the same condition moved from +54 % to +2 points once they were
        # excluded).
        "mean_drop_points": float(drop.mean()),
        "drop_points_ci95": [drop_low, drop_high],
        "drop_points_excludes_zero": bool(drop_low > 0.0 or drop_high < 0.0),
        "relative_drop_percent_mean": float(relative.mean()),
        "relative_drop_percent_ci95": [low, high],
        "relative_drop_percent_of_means": float(
            100.0 * drop.mean() / max(float(reference.mean()), 1e-8)),
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
        "median_delta": float(np.median(delta)),
        "delta_std": float(delta.std(ddof=1)) if len(delta) > 1 else float("nan"),
        "n_sequences": int(len(delta)),
    }
    if len(delta) > 1 and np.any(delta != 0):
        t_stat, t_p = stats.ttest_rel(condition, reference)
        out["paired_t"] = float(t_stat)
        out["paired_t_p"] = float(t_p)
        try:
            w_stat, w_p = stats.wilcoxon(condition, reference)
            out["wilcoxon"] = float(w_stat)
            out["wilcoxon_p"] = float(w_p)
        except ValueError:
            pass
    else:
        out["paired_t"] = float("nan")
        out["paired_t_p"] = float("nan")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True,
                        help="label=directory, repeatable")
    parser.add_argument("--reference", default="clean")
    parser.add_argument("--compare", nargs=2, action="append", default=[],
                        metavar=("LABEL_A", "LABEL_B"))
    parser.add_argument("--interaction", nargs=2, action="append", default=[],
                        metavar=("LABEL_A", "LABEL_B"),
                        help="difference in differences: is the clean-to-corrupt *drop* "
                             "smaller for run A than for run B?  Negative means A degrades "
                             "less, which is the actual robustness claim")
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--tracking-threshold", type=float, default=0.2,
                        help="the reference arm counts as tracking a sequence when its SR "
                             "exceeds this; the paired statistics are also reported on that "
                             "subset, because an already-lost sequence cannot get worse")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--strata-manifest", default=None,
                        help="manifest.json of the stratified split; adds a per-stratum "
                             "paired table, which is the reason the split is stratified")
    parser.add_argument("--screen-run", default=None,
                        help="label of the run the pre-registered strength screen is applied "
                             "to (the frozen no-decoder arm); prints pass/fail per condition")
    args = parser.parse_args()

    strata: Dict[str, str] = {}
    if args.strata_manifest:
        manifest = json.loads(Path(args.strata_manifest).read_text())
        strata = {row["sequence"]: row["stratum"] for row in manifest["sequences"]}
        if not strata:
            raise SystemExit(f"{args.strata_manifest} lists no sequences")

    runs: Dict[str, Dict[str, Dict]] = {}
    for spec in args.run:
        label, _, directory = spec.partition("=")
        if not directory:
            raise SystemExit(f"--run expects label=directory, got {spec!r}")
        runs[label] = read_run(Path(directory))

    rng = np.random.default_rng(args.seed)
    report: Dict[str, object] = {"reference": args.reference, "runs": {}}

    for label, conditions in runs.items():
        if args.reference not in conditions:
            raise SystemExit(f"run {label!r} has no {args.reference!r} condition")
        reference_metrics = conditions[args.reference]
        entry: Dict[str, object] = {
            "reference": {key: reference_metrics.get(key) for key in METRIC_KEYS},
            "n_reference_sequences": reference_metrics.get("n_sequences"),
            "conditions": {},
        }
        names: List[str] = []
        p_values: List[float] = []
        print(f"\n=== {label} (reference: {args.reference}, "
              f"{reference_metrics.get('n_sequences')} sequences) ===")
        print(f"{'condition':<28} {'n':>4} {'clean':>7} {'cond':>7} {'drop pts':>9} "
              f"{'95% CI (pts)':>18} {'rel%':>8} {'p(t)':>9}")
        for name, metrics in conditions.items():
            if name == args.reference:
                continue
            stats_row: Dict[str, float] = {}
            drop_points = float("nan")
            interval = (float("nan"), float("nan"))
            relative = float("nan")
            p_value = float("nan")
            for metric in METRIC_KEYS:
                reference_values = per_sequence(reference_metrics, metric)
                condition_values = per_sequence(metrics, metric)
                shared = sorted(set(reference_values) & set(condition_values))
                if len(shared) < 2:
                    stats_row[metric] = {"n_sequences": len(shared)}
                    continue
                reference = np.array([reference_values[s] for s in shared])
                condition = np.array([condition_values[s] for s in shared])
                # A sequence the evaluator could not score (NaN) must be dropped as a
                # *pair*: an unpaired mean would silently mix different sequence sets.
                finite = np.isfinite(reference) & np.isfinite(condition)
                dropped = int((~finite).sum())
                reference, condition = reference[finite], condition[finite]
                if len(reference) < 2:
                    stats_row[metric] = {"n_sequences": len(reference),
                                         "n_dropped_nonfinite": dropped}
                    continue
                metric_stats = paired_stats(reference, condition, rng, args.draws)
                if dropped:
                    metric_stats["n_dropped_nonfinite"] = dropped
                if metric == "sr":
                    # Secondary, pre-declared view: a sequence the *reference* arm never
                    # tracked (SR below the threshold) has no room to get worse, so the
                    # full-set delta understates the corruption effect on the sequences
                    # that are actually being tracked.  Both numbers are reported; the
                    # full set stays the primary comparison.
                    tracking = reference > args.tracking_threshold
                    metric_stats["n_reference_tracking"] = int(tracking.sum())
                    if int(tracking.sum()) >= 2:
                        metric_stats["sr_reference_tracking"] = paired_stats(
                            reference[tracking], condition[tracking], rng, args.draws)
                stats_row[metric] = metric_stats
                if metric == "sr":
                    drop_points = metric_stats["mean_drop_points"] * 100.0
                    low, high = metric_stats["drop_points_ci95"]
                    interval = (low * 100.0, high * 100.0)
                    relative = metric_stats["relative_drop_percent_mean"]
                    p_value = metric_stats.get("paired_t_p", float("nan"))
            stats_row["pooled"] = {key: metrics.get(key) for key in POOLED_KEYS}
            stats_row["n_sequences_evaluated"] = metrics.get("n_sequences")
            stats_row["n_skipped"] = metrics.get("n_skipped")
            entry["conditions"][name] = stats_row
            names.append(name)
            p_values.append(p_value if np.isfinite(p_value) else 1.0)
            print(f"{name:<28} {stats_row['sr'].get('n_sequences', 0):>4} "
                  f"{stats_row['sr'].get('mean_reference', float('nan')) * 100:>7.2f} "
                  f"{stats_row['sr'].get('mean_condition', float('nan')) * 100:>7.2f} "
                  f"{drop_points:>9.2f} "
                  f"[{interval[0]:>7.2f}, {interval[1]:>7.2f}] {relative:>8.2f} "
                  f"{p_value:>9.4f}")
            subset = stats_row["sr"].get("sr_reference_tracking")
            if subset:
                print(f"{'  (reference-tracking subset)':<28} {subset['n_sequences']:>4} "
                      f"{subset['mean_reference'] * 100:>7.2f} "
                      f"{subset['mean_condition'] * 100:>7.2f} "
                      f"{subset['mean_drop_points'] * 100:>9.2f} "
                      f"[{subset['drop_points_ci95'][0] * 100:>7.2f}, "
                      f"{subset['drop_points_ci95'][1] * 100:>7.2f}] "
                      f"{subset['relative_drop_percent_mean']:>8.2f} "
                      f"{subset.get('paired_t_p', float('nan')):>9.4f}")
        for name, adjusted in zip(names, holm(p_values)):
            entry["conditions"][name]["sr"]["paired_t_p_holm"] = float(adjusted)

        if strata:
            # The split exists to say *which* challenge the corruption interacts with, so the
            # per-stratum paired delta is reported rather than only the pooled one.
            entry["strata"] = {}
            reference_values = per_sequence(reference_metrics, "sr")
            print(f"\n--- {label}: paired SR drop (points) by stratum ---")
            header = "  ".join(f"{s[:16]:>16}" for s in sorted(set(strata.values())))
            print(f"{'condition':<28} {header}")
            for name, metrics in conditions.items():
                if name == args.reference:
                    continue
                condition_values = per_sequence(metrics, "sr")
                per_stratum: Dict[str, Dict[str, float]] = {}
                cells = []
                for stratum in sorted(set(strata.values())):
                    names_in = [s for s in reference_values
                                if strata.get(s) == stratum and s in condition_values]
                    if len(names_in) < 2:
                        cells.append(f"{'-':>16}")
                        continue
                    ref = np.array([reference_values[s] for s in names_in])
                    cond = np.array([condition_values[s] for s in names_in])
                    row = {
                        "n_sequences": len(names_in),
                        "mean_reference": float(ref.mean()),
                        "mean_condition": float(cond.mean()),
                        "mean_drop_points": float((ref - cond).mean()),
                    }
                    per_stratum[stratum] = row
                    cells.append(f"{row['mean_drop_points'] * 100:>+16.2f}")
                entry["strata"][name] = per_stratum
                print(f"{name:<28} " + "  ".join(cells))
        report["runs"][label] = entry

    if args.compare:
        report["comparisons"] = []
        for label_a, label_b in args.compare:
            if label_a not in runs or label_b not in runs:
                raise SystemExit(f"unknown run in --compare {label_a} {label_b}")
            shared_conditions = sorted(set(runs[label_a]) & set(runs[label_b]))
            print(f"\n=== paired comparison {label_a} - {label_b} ===")
            print(f"{'condition':<32} {'n':>4} {'A-B':>8} {'95% CI':>18} {'p(t)':>9}")
            for name in shared_conditions:
                values_a = per_sequence(runs[label_a][name], "sr")
                values_b = per_sequence(runs[label_b][name], "sr")
                shared = sorted(set(values_a) & set(values_b))
                if len(shared) < 2:
                    continue
                a = np.array([values_a[s] for s in shared])
                b = np.array([values_b[s] for s in shared])
                delta = a - b
                low, high = bootstrap_interval(delta, rng, args.draws)
                t_p = (float(stats.ttest_rel(a, b).pvalue)
                       if np.any(delta != 0) else float("nan"))
                row = {
                    "n_sequences": len(shared),
                    "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                    "mean_delta": float(delta.mean()),
                    "delta_ci95": [low, high],
                    "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
                    "paired_t_p": t_p,
                }
                report["comparisons"].append({"a": label_a, "b": label_b,
                                              "condition": name, **row})
                print(f"{name:<32} {len(shared):>4} {delta.mean() * 100:>8.2f} "
                      f"[{low * 100:>7.2f}, {high * 100:>7.2f}] {t_p:>9.4f}")

    if args.screen_run:
        label = args.screen_run
        if label not in report["runs"]:
            raise SystemExit(f"--screen-run {label!r} is not one of the runs")
        rows = report["runs"][label]["conditions"]
        print(f"\n=== pre-registered strength screen on {label} "
              f"(relative drop {SCREEN_RELATIVE_DROP[0]:.0f}-{SCREEN_RELATIVE_DROP[1]:.0f} %, "
              f"retained {SCREEN_RETAINED[0]:.0f}-{SCREEN_RETAINED[1]:.0f} % of clean SR) ===")
        print(f"{'condition':<28} {'rel%':>8} {'retained%':>10} {'CI excl 0':>9} "
              f"{'screen':>11}")
        for name, row in sorted(rows.items()):
            sr = row.get("sr", {})
            if "relative_drop_percent_mean" not in sr:
                continue
            relative = sr["relative_drop_percent_mean"]
            retained = 100.0 * sr["mean_condition"] / max(sr["mean_reference"], 1e-8)
            excludes = sr["drop_points_excludes_zero"]
            inside = (SCREEN_RELATIVE_DROP[0] <= relative <= SCREEN_RELATIVE_DROP[1]
                      and SCREEN_RETAINED[0] <= retained <= SCREEN_RETAINED[1])
            if not excludes:
                verdict = "n.s."
            elif relative < 0.0:
                verdict = "helps"          # corruption improved tracking: not a degradation
            elif inside:
                verdict = "pass"
            elif relative < SCREEN_RELATIVE_DROP[0]:
                verdict = "too weak"
            elif relative > SCREEN_RELATIVE_DROP[1]:
                verdict = "too strong"
            else:
                verdict = "retained out of band"
            print(f"{name:<28} {relative:>8.2f} {retained:>10.2f} "
                  f"{str(excludes):>9} {verdict:>11}")
        print("Report the whole curve: a level chosen because one arm looks best there is not "
              "a calibration.")

    if args.interaction:
        report["interactions"] = []
        for label_a, label_b in args.interaction:
            if label_a not in runs or label_b not in runs:
                raise SystemExit(f"unknown run in --interaction {label_a} {label_b}")
            if args.reference not in runs[label_a] or args.reference not in runs[label_b]:
                raise SystemExit("--interaction needs the reference condition in both runs")
            print(f"\n=== difference in differences: drop({label_a}) - drop({label_b}) ===")
            print("negative = the first arm's corruption-induced drop is smaller")
            print(f"{'condition':<28} {'n':>4} {'d_A':>8} {'d_B':>8} {'A-B':>8} "
                  f"{'95% CI (pts)':>18} {'p(t)':>9}")
            for name in sorted(set(runs[label_a]) & set(runs[label_b])):
                if name == args.reference:
                    continue
                ref_a = per_sequence(runs[label_a][args.reference], "sr")
                ref_b = per_sequence(runs[label_b][args.reference], "sr")
                cond_a = per_sequence(runs[label_a][name], "sr")
                cond_b = per_sequence(runs[label_b][name], "sr")
                shared = sorted(set(ref_a) & set(ref_b) & set(cond_a) & set(cond_b))
                if len(shared) < 2:
                    continue
                drop_a = np.array([ref_a[s] - cond_a[s] for s in shared])
                drop_b = np.array([ref_b[s] - cond_b[s] for s in shared])
                finite = np.isfinite(drop_a) & np.isfinite(drop_b)
                drop_a, drop_b = drop_a[finite], drop_b[finite]
                if len(drop_a) < 2:
                    continue
                diff = drop_a - drop_b
                row = difference_in_differences(drop_a, drop_b, rng, args.draws)
                low, high = row["difference_ci95"]
                t_p = row["paired_t_p"]
                report["interactions"].append({"a": label_a, "b": label_b,
                                               "condition": name, **row})
                print(f"{name:<28} {len(shared):>4} {drop_a.mean() * 100:>8.2f} "
                      f"{drop_b.mean() * 100:>8.2f} {diff.mean() * 100:>8.2f} "
                      f"[{low * 100:>7.2f}, {high * 100:>7.2f}] {t_p:>9.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")
    print("\nIntervals resample sequences (the independent unit), never frames; the "
          "corruption sample is fixed per sequence, so pairs are matched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
