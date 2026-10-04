#!/usr/bin/env python
"""Fixed-crop replay: is a corruption effect the model's response, or trajectory divergence?

A closed-loop tracker cannot attribute its own failures: frame 40 is cropped around the
prediction of frame 39, so one bad frame changes every later crop and "the corruption hurt"
and "the trajectory diverged" become the same sentence.

This tool runs a reference arm (by default the *clean* run on the same condition-free path) and
keeps its per-frame boxes; every condition is then re-run with ``fixed_boxes`` set to that
reference trajectory, so all conditions see **identical crops** and a per-frame difference is
the model's response to that frame.  Reported per condition:

* ``free``  -- mean IoU of the normal closed-loop run;
* ``fixed`` -- mean IoU when replayed on the reference crops;
* ``single_step`` -- mean IoU over the first ``--single-step-frames`` replayed frames, the only
  window in which every condition is guaranteed to share the entire history.

If the clean-vs-corrupt gap closes under fixed crops, the effect is trajectory divergence; if it
survives, it is a per-frame response and worth attributing further.

**Stage 2 -- path interventions.**  For every frame the corrupted pass is composed with
quantities cached from a *clean* pass of the same frame (the same crops), substituted at module
boundaries so the model code is untouched:

* ``control``: ``model.reliability`` and ``model.gating`` return the clean run's values, so the
  corrupt run keeps its features but loses the corrupt-specific control signals (r, severity
  gates);
* ``feature``: ``model.fusion`` receives the clean run's corrected **TIR** tokens and TIR FPN
  taps while every control quantity stays corrupt -- this asks whether the TIR *content* is
  what changes the result;
* ``both``: both substitutions.

Each intervention is therefore a single forward on the same crop with a documented cache, run
through ``Trainer.infer_sequence(forward_fn=...)`` so it uses the real tracker loop.

Usage::

    python tools/trajectory_replay.py --checkpoint outputs/ab_full/final.pth \
        --sequences 5 --frames 60 --target tir --ratio 0.2 \
        --out outputs/trajectory_replay_ab_full_tir.json
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codetrack.utils.runtime import CPU_THREAD_SETTINGS  # noqa: E402,F401

import numpy as np  # noqa: E402
import torch  # noqa: E402

from codetrack.engine.evaluator import summarize_recovery  # noqa: E402
from codetrack.metrics import _box_iou  # noqa: E402
from codetrack.utils.checkpoint import load_checkpoint  # noqa: E402
from codetrack.utils.config import load_config  # noqa: E402
from recovery_probe import list_sequences  # noqa: E402


def path_substitution(model):
    """Context manager capturing the clean pass and substituting it in a later pass.

    Returns ``(capture, apply)`` where ``capture(...)`` runs a forward and records the clean
    ``reliability`` output, the severity gates, the corrected TIR tokens and the TIR FPN taps,
    and ``apply(mode)`` is a context manager that injects those caches into a later forward:

    * ``control`` -- reliability and gating come from the cache;
    * ``feature`` -- fusion receives the cached TIR tokens/taps;
    * ``both``     -- both.

    Substituting from *outside* the per-frame forward (as an earlier draft did) would put one
    frame's tensors on another frame's input, which measures nothing; this version is called
    once per frame with that frame's cache.
    """
    cache: Dict[str, object] = {}

    def capture_run(forward, *args, **kwargs):
        original_reliability = model.reliability.forward
        original_gating = model.gating.forward
        original_decoder = model.decoder.forward
        original_fusion = model.fusion.forward

        def reliability(*a, **kw):
            out = original_reliability(*a, **kw)
            cache["reliability"] = {k: v.detach() for k, v in out.items()}
            return out

        def gating(*a, **kw):
            gate_rgb, gate_tir = original_gating(*a, **kw)
            cache["gate"] = (gate_rgb.detach(), gate_tir.detach())
            return gate_rgb, gate_tir

        def decoder(*a, **kw):
            out = original_decoder(*a, **kw)
            cache["corrected_tir"] = out["corrected_tir"].detach()
            return out

        def fusion(corrected_rgb, corrected_tir, residual_rgb, residual_tir, taps_r, taps_t):
            cache["corrected_rgb"] = corrected_rgb.detach()
            cache["taps_t"] = {k: v.detach() for k, v in taps_t.items()}
            return original_fusion(corrected_rgb, corrected_tir, residual_rgb, residual_tir,
                                   taps_r, taps_t)

        model.reliability.forward = reliability
        model.gating.forward = gating
        model.decoder.forward = decoder
        model.fusion.forward = fusion
        try:
            with torch.no_grad():
                return forward(*args, **kwargs)
        finally:
            model.reliability.forward = original_reliability
            model.gating.forward = original_gating
            model.decoder.forward = original_decoder
            model.fusion.forward = original_fusion

    @contextmanager
    def apply(mode: str):
        originals = []
        if mode in ("control", "both"):
            reliability_out = cache["reliability"]
            gate_rgb, gate_tir = cache["gate"]
            originals.append((model.reliability, model.reliability.forward,
                              lambda *a, **kw: reliability_out))
            originals.append((model.gating, model.gating.forward,
                              lambda *a, **kw: (gate_rgb, gate_tir)))
        if mode in ("feature", "both"):
            cached_tir = cache["corrected_tir"]
            cached_taps = cache["taps_t"]
            original_fusion = model.fusion.forward

            def fusion(corrected_rgb, corrected_tir, residual_rgb, residual_tir,
                       taps_r, taps_t):
                return original_fusion(corrected_rgb, cached_tir, residual_rgb, residual_tir,
                                       taps_r, cached_taps)

            originals.append((model.fusion, model.fusion.forward, fusion))
        for module, _, replacement in originals:
            module.forward = replacement
        try:
            yield
        finally:
            for module, original, _ in originals:
                module.forward = original

    return cache, capture_run, apply


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/lasher_vitb_corrupt.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", default="/root/autodl-tmp/lab/dataset/LasHeR")
    parser.add_argument("--subset", default="testingset")
    parser.add_argument("--sequence-list", default=None,
                        help="optional list file; otherwise the first --sequences of the subset")
    parser.add_argument("--sequences", type=int, default=5)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--single-step-frames", type=int, default=5)
    parser.add_argument("--token", default="tok_block_erase")
    parser.add_argument("--target", default="tir", choices=["rgb", "tir", "both"])
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--severity", type=float, default=0.4)
    parser.add_argument("--fixed-gt", action="store_true",
                        help="with --recovery, also replay the corrupted arm on the "
                             "ground-truth trajectory.  Training always crops around the target, "
                             "while a closed loop crops around its own drifting prediction; if "
                             "the branch only repairs under centred crops, this shows it")
    parser.add_argument("--recovery", action="store_true",
                        help="also collect the per-frame recovery diagnostics of the clean, "
                             "free-crop and fixed-crop arms and report recovery_gain.  The "
                             "training loss logs `gain 0.0000 act 0.00` (the repair constraint "
                             "satisfied on training batches) while evaluation reports a negative "
                             "recovery_gain; if the negative value is drift-driven it should "
                             "move towards the training-time value when the crops are held fixed")
    parser.add_argument("--seed", type=int, default=0,
                        help="reseeded before *every* arm of a sequence.  The token corruption "
                             "draws from the global torch RNG, so an arm that runs extra "
                             "forwards would otherwise shift the mask stream of every later "
                             "arm and of every later sequence -- the arms would no longer see "
                             "the same corruption")
    parser.add_argument("--intervene", default="feature,control",
                        help="comma separated path substitutions to run on the fixed reference "
                             "crops: feature (clean TIR tokens/taps into the fusion), control "
                             "(clean reliability + severity gates), both, or '' to disable")
    parser.add_argument("--out", default="outputs/trajectory_replay.json")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    from codetrack.engine.trainer import Trainer  # noqa: E402

    trainer = Trainer(cfg, output_dir=Path(args.out).parent / "replay")
    report = load_checkpoint(args.checkpoint, trainer.model,
                             map_location=str(trainer.device))
    trainer.logger.info("loaded checkpoint %s (epoch %s)", args.checkpoint, report["epoch"])
    trainer.model.eval()

    root = Path(args.root)
    if args.sequence_list:
        sequences = [line.strip() for line in Path(args.sequence_list).read_text().splitlines()
                     if line.strip()][:args.sequences]
    else:
        sequences = list_sequences(root, args.subset, args.sequences)

    modes = [value.strip() for value in args.intervene.split(",") if value.strip()]

    def run_arm(**kwargs):
        """One arm of one sequence, always from the same RNG state (see --seed)."""
        torch.manual_seed(args.seed)
        return trainer.infer_sequence(root, args.subset, kwargs.pop("sequence"),
                                      **kwargs)

    corruption = {"enabled": True, "token": [args.token], "rgb": [], "tir": [],
                  "cross_modal": False, "ratio": args.ratio, "severity": args.severity,
                  "target": args.target}
    rows: List[Dict[str, object]] = []
    for sequence in sequences:
        clean = run_arm(sequence=sequence, max_frames=args.frames, collect=False)
        if len(clean["pred"]) < 2:
            rows.append({"sequence": sequence, "skipped": "no frames"})
            continue
        reference, gt = clean["pred"], clean["gt"]
        free = run_arm(sequence=sequence, max_frames=args.frames,
                       corruption=corruption, collect=False)
        replay = run_arm(sequence=sequence, max_frames=args.frames,
                         corruption=corruption, collect=False, fixed_boxes=reference)
        head = args.single_step_frames
        # Did the replay actually see the same crops as the free run?  This is the assumption the
        # whole fixed-crop comparison rests on, so it is measured from the crops the loop recorded
        # rather than assumed.  The first version compared frame-0 *predictions* and ORed in
        # ``len(gt) > 0``, which made it true for every non-empty sequence and hid a one-frame
        # misalignment of the reference schedule (see crop_box_for_frame).
        free_crops, replay_crops = free.get("crops"), replay.get("crops")
        matched_frames = 0
        if free_crops is not None and replay_crops is not None and len(free_crops):
            common = min(len(free_crops), len(replay_crops))
            matched_frames = int(np.all(np.isclose(free_crops[:common], replay_crops[:common],
                                                   atol=1e-6), axis=1).sum())
        row = {
            "sequence": sequence,
            "interventions": {},
            "recovery": {},
            "frames": int(len(gt)),
            "clean_iou": float(_box_iou(reference, gt).mean()),
            "free_iou": float(_box_iou(free["pred"], gt).mean()),
            "fixed_crop_iou": float(_box_iou(replay["pred"], gt).mean()),
            "clean_single_step": float(_box_iou(reference[:head], gt[:head]).mean()),
            "free_single_step": float(_box_iou(free["pred"][:head], gt[:head]).mean()),
            "fixed_crop_single_step": float(_box_iou(replay["pred"][:head], gt[:head]).mean()),
            # frame 0 only: every arm starts from the same annotation-driven crop, so this must be
            # true and a failure means the replay is not the comparison it claims to be
            "crop_identical": bool(
                free_crops is not None and replay_crops is not None and len(free_crops)
                and np.allclose(free_crops[0], replay_crops[0], atol=1e-6)),
            # how far the closed loops have diverged: frames whose crops still agree
            "crop_identical_frames": matched_frames,
        }
        if args.recovery:
            detail = {}
            arms_to_run = [("clean", {}), ("free", {}), ("fixed", {"fixed_boxes": reference})]
            if args.fixed_gt:
                arms_to_run.append(("ground_truth_crop", {"fixed_boxes": gt}))
            for arm_name, kwargs in arms_to_run:
                arm_corruption = None if arm_name == "clean" else corruption
                run = run_arm(sequence=sequence, max_frames=args.frames, collect=True,
                              corruption=arm_corruption, **kwargs)
                diag = run.get("diagnostics") or {}
                if not diag:
                    continue
                merged = {key: [x for item in diag[key] for x in item]
                          for key in ("e_before", "e_after", "e_clean") if key in diag}
                if len(merged) == 3:
                    detail[arm_name] = summarize_recovery(**merged)
            row["recovery"] = detail

        for mode in modes:
            cache, capture_run, apply_substitution = path_substitution(trainer.model)

            def wrapper(tpl_rgb, search_rgb, tpl_tir, search_tir, corruption=None,
                        clean_teacher=False, _capture=capture_run, _apply=apply_substitution,
                        _mode=mode):
                # one clean pass on the *same crops* provides the caches for this frame
                _capture(trainer.model, tpl_rgb, search_rgb, tpl_tir, search_tir,
                         corruption=None)
                with _apply(_mode):
                    return trainer.model(tpl_rgb, search_rgb, tpl_tir, search_tir,
                                         corruption=corruption, clean_teacher=clean_teacher)

            mixed = run_arm(sequence=sequence, max_frames=args.frames,
                            corruption=corruption, collect=False, fixed_boxes=reference,
                            forward_fn=wrapper)
            row["interventions"][mode] = {
                "mean_iou": float(_box_iou(mixed["pred"], gt).mean()),
                "single_step_iou": float(_box_iou(mixed["pred"][:head], gt[:head]).mean()),
            }
        rows.append(row)
        trainer.logger.info(
            "%s: clean %.3f | corrupt free %.3f fixed-crop %.3f | single-step clean %.3f "
            "free %.3f fixed %.3f | interventions %s", sequence, row["clean_iou"],
            row["free_iou"], row["fixed_crop_iou"], row["clean_single_step"],
            row["free_single_step"], row["fixed_crop_single_step"],
            {k: round(v["mean_iou"], 3) for k, v in row["interventions"].items()})

    valid = [row for row in rows if "clean_iou" in row]

    def mean(key: str) -> float:
        values = [row[key] for row in valid]
        return float(np.mean(values)) if values else float("nan")

    summary = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "overrides": args.override,
        "token": args.token,
        "target": args.target,
        "ratio": args.ratio,
        "severity": args.severity,
        "seed": args.seed,
        "sequences": len(valid),
        "frames": args.frames,
        "single_step_frames": args.single_step_frames,
        "interventions": {
            mode: {
                "mean_iou": float(np.mean([row["interventions"][mode]["mean_iou"]
                                           for row in valid if mode in row["interventions"]])),
                "single_step_iou": float(np.mean(
                    [row["interventions"][mode]["single_step_iou"] for row in valid
                     if mode in row["interventions"]])),
            } for mode in modes
        } if modes else {},
        "mean_iou": {
            "clean": mean("clean_iou"),
            "corrupt_free": mean("free_iou"),
            "corrupt_fixed_crop": mean("fixed_crop_iou"),
            "clean_single_step": mean("clean_single_step"),
            "corrupt_free_single_step": mean("free_single_step"),
            "corrupt_fixed_single_step": mean("fixed_crop_single_step"),
        },
        "rows": rows,
        "note": ("`free` is the closed-loop run; `fixed_crop` replays the clean arm's crops, so "
                 "all conditions share the history.  A gap that closes under fixed crops is "
                 "trajectory divergence; a gap that survives is a per-frame response.  Stage-2 "
                 "path interventions need per-frame hooks inside Trainer.infer_sequence and are "
                 "not implemented here."),
    }
    values = summary["mean_iou"]
    if args.recovery:
        arms = sorted({arm for row in valid for arm in (row.get("recovery") or {})})
        recovery_summary = {}
        for arm in arms:
            gains = [row["recovery"][arm].get("recovery_gain") for row in valid
                     if arm in (row.get("recovery") or {})]
            gains = [value for value in gains if value is not None and np.isfinite(value)]
            if not gains:
                # the uncorrupted arm has no damaged tokens, so its recovery is undefined
                continue
                continue
            recovery_summary[arm] = {
                "n": len(gains),
                "mean_recovery_gain": float(np.mean(gains)),
                "median_recovery_gain": float(np.median(gains)),
                "fraction_positive": float(np.mean([value > 0 for value in gains])),
            }
        summary["recovery"] = recovery_summary
        print("\nrecovery_gain per arm (positive = the branch reduced the error on damaged "
              "tokens):")
        for arm, arm_stats in recovery_summary.items():
            print(f"  {arm:<6} n={arm_stats['n']:<3} mean "
                  f"{arm_stats['mean_recovery_gain']:+.4f} median "
                  f"{arm_stats['median_recovery_gain']:+.4f} positive on "
                  f"{arm_stats['fraction_positive'] * 100:.0f}% of sequences")

    if modes:
        for mode in modes:
            values_mode = summary["interventions"][mode]
            print(f"  intervention {mode:<8} fixed-crop mean IoU "
                  f"{values_mode['mean_iou']:.4f} single-step "
                  f"{values_mode['single_step_iou']:.4f}")
    print(f"sequences={len(valid)} frames={args.frames} target={args.target} | clean "
          f"{values['clean']:.4f} corrupt-free {values['corrupt_free']:.4f} corrupt-fixed-crop "
          f"{values['corrupt_fixed_crop']:.4f} | single-step clean "
          f"{values['clean_single_step']:.4f} free {values['corrupt_free_single_step']:.4f} "
          f"fixed {values['corrupt_fixed_single_step']:.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
