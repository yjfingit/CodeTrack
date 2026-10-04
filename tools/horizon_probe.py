#!/usr/bin/env python
"""Measure *when* the tracker loses a sequence, not just how often it succeeds.

``SR (AUC)`` is an average over the whole evaluation window, so a tracker that tracks
perfectly for 60 frames and then drifts away for 140 scores about the same as one that
tracks at IoU 0.45 for 200.  The long-horizon validation split made that ambiguity
load-bearing: on the 60-sequence split the clean checkpoint averages 0.34 SR over 200
frames, and any clean-vs-corrupt comparison on top of it needs to know whether the clean
arm is still tracking at all.

This tool runs the tracker over full sequences and reports, per sequence:

* the per-frame IoU trace (downsampled) and ``sr`` / ``pr`` for reference;
* ``loss_frame`` -- the first frame whose 20-frame rolling mean IoU stays below 0.5 for
  the rest of the sequence (``None`` if the sequence never loses the target);
* ``tracked_fraction`` -- the fraction of frames up to ``loss_frame`` (1.0 when never lost);
* ``iou_mean_first`` / ``iou_mean_last`` -- first and last 20-frame windows.

The point is a *distribution of loss time*, which is what a corruption comparison has to
be read against: corruption can only change the loss time, not the score of frames that
were already lost.

Usage::

    python tools/horizon_probe.py --checkpoint outputs/ab_full/final.pth \
        --sequence-list outputs/validation_split_v1/sequences.txt --frames 200 \
        --out outputs/horizon_probe_ab_full.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codetrack.utils.runtime import CPU_THREAD_SETTINGS  # noqa: E402,F401

import numpy as np

from codetrack.metrics import _box_iou, precision_at, success_auc  # noqa: E402
from codetrack.utils.checkpoint import load_checkpoint  # noqa: E402
from codetrack.utils.config import load_config  # noqa: E402


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        window = max(1, len(values))
    kernel = np.ones(window) / float(window)
    return np.convolve(values, kernel, mode="valid")


def loss_frame(iou: np.ndarray, window: int = 20, threshold: float = 0.5) -> Optional[int]:
    """First frame after which the rolling mean IoU never returns above ``threshold``.

    Descriptive only: it looks at the *whole* trace and at the total horizon, so it is not a
    first-failure event (see :func:`first_failure` for the forward-decidable version).
    """
    smoothed = rolling_mean(iou, window)
    below = smoothed < threshold
    for start in range(len(below)):
        if below[start:].all():
            return start + window // 2
    return None


def first_failure(iou: np.ndarray, threshold: float = 0.5, patience: int = 20
                  ) -> Optional[int]:
    """First frame of a run of ``patience`` consecutive frames below ``threshold``.

    Forward-decidable: the decision at frame ``t`` depends only on frames ``<= t``, so it is a
    real event a tracker could detect online, and it does not need the rest of the horizon.
    """
    run = 0
    for index, value in enumerate(iou):
        run = run + 1 if float(value) < threshold else 0
        if run >= patience:
            return index - patience + 1
    return None


def failure_confirmed_frame(failure: Optional[int], patience: int) -> Optional[int]:
    """The first frame at which the failure is *knowable* online.

    :func:`first_failure` deliberately reports the **start** of the failing run, because that is the
    interpretable event time.  But a tracker can only act on it once ``patience`` consecutive
    frames have been observed, so the two differ by ``patience - 1`` frames -- 19 frames under the
    20-frame rule used throughout ``docs/results.md``.  Reporting only the start frame next to the
    phrase "forward-decidable" overstates how early the event is available, so both are stored.
    """
    if failure is None:
        return None
    return int(failure) + int(patience) - 1


def rmst(times: np.ndarray, events: np.ndarray, tau: int) -> float:
    """Restricted mean survival time: mean frames tracked, censored at ``tau``.

    ``times`` are time-to-event values; ``events`` marks whether the event was observed (1) or the
    sequence was still tracked at the horizon (0, right-censored).  Kaplan-Meier areas are used
    rather than a plain mean so that sequences which never failed are not silently treated as if
    they had failed exactly at the horizon.

    Why this endpoint exists: SR and mean IoU both collapse a trajectory into one number, and
    section 6.17.1 measured that mean IoU is *not* the lower-variance option.  RMST keeps the
    time dimension, is right-censoring aware, and is the endpoint a "delays failure" claim is
    actually about.  It is a **mechanistic secondary** endpoint, never a replacement for the
    pre-registered SR contrast.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float)
    if times.size == 0:
        return float("nan")
    horizon = float(min(tau, times.max()))
    if horizon <= 0:
        return 0.0
    # Ties are grouped: frame indices are integers, so several sequences routinely fail at the
    # same frame, and treating them one at a time would give each a different risk-set size.
    order = np.argsort(times)
    times, events = times[order], events[order]
    area = 0.0
    previous = 0.0
    survival = 1.0
    at_risk = times.size
    index = 0
    while index < times.size:
        time = float(times[index])
        if time > horizon:
            break
        stop = index
        while stop < times.size and float(times[stop]) == time:
            stop += 1
        area += survival * (time - previous)
        previous = time
        events_here = float(events[index:stop].sum())
        if events_here and at_risk > 0:
            survival *= (1.0 - events_here / at_risk)
        at_risk -= stop - index
        index = stop
    area += survival * (horizon - previous)
    return float(area)


def recovered_after(iou: np.ndarray, failure: Optional[int], threshold: float = 0.5,
                    patience: int = 20) -> Optional[bool]:
    """Did a run of ``patience`` consecutive frames at or above threshold follow the failure?"""
    if failure is None:
        return None
    run = 0
    for value in iou[failure + patience:]:
        run = run + 1 if float(value) >= threshold else 0
        if run >= patience:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/lasher_vitb_corrupt.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--subset", default="testingset")
    parser.add_argument("--sequence-list", default=None)
    parser.add_argument("--sequences", type=int, default=10,
                        help="how many sequences to probe when no list is given")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--patience", type=int, default=20,
                        help="consecutive frames below the threshold that count as a failure; "
                             "this is the forward-decidable definition")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--rmst-tau", type=int, default=200,
                        help="horizon of the restricted mean survival time endpoint; must be "
                             "within the follow-up both arms actually have")
    parser.add_argument("--out", default="outputs/horizon_probe.json")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    from codetrack.engine.trainer import Trainer  # noqa: E402

    trainer = Trainer(cfg, output_dir=Path(args.out).parent / "horizonprobe")
    report = load_checkpoint(args.checkpoint, trainer.model,
                            map_location=str(trainer.device))
    trainer.logger.info("loaded checkpoint %s (epoch %s)", args.checkpoint, report["epoch"])
    trainer.model.eval()

    root = Path(args.root or cfg.get("data", {}).get("root", "data/LasHeR"))
    if args.sequence_list:
        sequences = [line.strip() for line in Path(args.sequence_list).read_text()
                     .splitlines() if line.strip()]
    else:
        list_name = ("testingsetList.txt" if args.subset.startswith("test")
                     else "trainingsetList.txt")
        sequences = [line.strip() for line in (root / list_name).read_text().splitlines()
                     if line.strip()][:args.sequences]

    rows: List[Dict[str, object]] = []
    for sequence in sequences:
        run = trainer.infer_sequence(root, args.subset, sequence, max_frames=args.frames)
        pred, gt = run["pred"], run["gt"]
        if len(pred) == 0:
            rows.append({"sequence": sequence, "skipped": True})
            continue
        iou = _box_iou(pred, gt)
        lost = loss_frame(iou, args.window, args.threshold)
        failure = first_failure(iou, args.threshold, args.patience)
        recovered = recovered_after(iou, failure, args.threshold, args.patience)
        window = min(args.window, len(iou))
        row = {
            "sequence": sequence,
            "frames": int(len(iou)),
            "sr": float(success_auc(pred, gt)),
            "pr": float(precision_at(pred, gt, 20.0)),
            "iou_mean": float(iou.mean()),
            "iou_mean_first": float(iou[:window].mean()),
            "iou_mean_last": float(iou[-window:].mean()),
            "loss_frame": lost,
            # the event *start*; interpretable, and what a "median failure frame" means
            "failure_frame": failure,
            # the frame at which the event is knowable online: start + patience - 1
            "failure_confirmed_frame": failure_confirmed_frame(failure, args.patience),
            "recovered_after_failure": recovered,
            "time_to_failure_censored": (int(len(iou)) if failure is None else int(failure)),
            "tracked_fraction": 1.0 if lost is None else float(lost) / len(iou),
            "iou_trace_every_10": [float(iou[i]) for i in range(0, len(iou), 10)],
        }
        rows.append(row)
        trainer.logger.info("%s: frames=%d sr=%.3f iou_mean=%.3f loss_frame=%s",
                            sequence, len(iou), row["sr"], row["iou_mean"], lost)

    valid = [row for row in rows if not row.get("skipped")]
    summary: Dict[str, object] = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "overrides": args.override,
        "sequence_list": args.sequence_list,
        "frames": args.frames,
        "window": args.window,
        "threshold": args.threshold,
        "n_sequences": len(valid),
        "sr_mean": float(np.mean([row["sr"] for row in valid])) if valid else float("nan"),
        "iou_mean": float(np.mean([row["iou_mean"] for row in valid])) if valid else float("nan"),
        "never_lost_fraction": (float(np.mean([row["loss_frame"] is None for row in valid]))
                               if valid else float("nan")),
        "loss_frame_median": (float(np.median([row["loss_frame"] for row in valid
                                               if row["loss_frame"] is not None]))
                              if any(row["loss_frame"] is not None for row in valid)
                              else None),
        # forward-decidable failure event: K consecutive frames below threshold, with the
        # un-failed sequences right-censored at the horizon
        "failure_patience": args.patience,
        "failure_fraction": (float(np.mean([row["failure_frame"] is not None for row in valid]))
                             if valid else float("nan")),
        "failure_frame_median": (float(np.median([row["failure_frame"] for row in valid
                                                  if row["failure_frame"] is not None]))
                                 if any(row["failure_frame"] is not None for row in valid)
                                 else None),
        # The same statistic on the *konwable* frame.  The two differ by patience - 1, and quoting
        # the start frame next to "forward-decidable" would overstate how early it is available.
        "failure_confirmed_frame_median": (
            float(np.median([row["failure_confirmed_frame"] for row in valid
                             if row["failure_confirmed_frame"] is not None]))
            if any(row["failure_confirmed_frame"] is not None for row in valid) else None),
        # RMST: mean frames tracked up to tau, right-censoring aware.  A mechanistic endpoint for
        # "does this arm delay failure", reported next to -- never instead of -- SR.
        "rmst_tau": int(args.rmst_tau),
        "rmst": rmst(np.array([row["time_to_failure_censored"] for row in valid], dtype=float),
                     np.array([row["failure_frame"] is not None for row in valid], dtype=float),
                     args.rmst_tau) if valid else float("nan"),
        "mean_time_to_failure_censored": (float(np.mean(
            [row["time_to_failure_censored"] for row in valid])) if valid else float("nan")),
        "recovery_rate_after_failure": (float(np.mean(
            [bool(row["recovered_after_failure"]) for row in valid
             if row["recovered_after_failure"] is not None]))
            if any(row["recovered_after_failure"] is not None for row in valid)
            else float("nan")),
        "loss_frame_quartiles": ([float(v) for v in np.quantile(
            [row["loss_frame"] for row in valid if row["loss_frame"] is not None],
            [0.25, 0.5, 0.75])]
            if any(row["loss_frame"] is not None for row in valid) else None),
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"failure (patience {args.patience} < {args.threshold}): "
          f"{summary['failure_fraction'] * 100:.1f}% of sequences | median failure frame "
          f"{summary['failure_frame_median']} (knowable at "
          f"{summary['failure_confirmed_frame_median']}) | RMST(tau={summary['rmst_tau']}) "
          f"{summary['rmst']:.1f} frames | recovery rate "
          f"{summary['recovery_rate_after_failure']:.2f}")
    print(f"{len(valid)} sequences | SR {summary['sr_mean']:.4f} | "
          f"mean IoU {summary['iou_mean']:.4f} | never lost "
          f"{summary['never_lost_fraction'] * 100:.1f}% | loss frame median "
          f"{summary['loss_frame_median']} | quartiles {summary['loss_frame_quartiles']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
