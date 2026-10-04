#!/usr/bin/env python
"""Is the training-time "repair constraint satisfied" signal an fp16 artefact?

Training runs the model under ``torch.autocast(float16)`` and the evaluation path runs it in fp32.
The training log reports ``gain 0.0000 act 0.00`` -- the hinge
``clamp(e_after - 0.9 * e_before, 0)`` satisfied, i.e. the branch reduces the damaged-token error
by more than 10 % -- while every evaluation, on the training corruption family, on held-out *and*
training sequences, with free *and* fixed crops, reports ``recovery_gain ~ -0.04`` (6.20 review).
Same helper, opposite verdict, so the remaining difference is the forward precision.

This script runs the *training* loss on real training batches twice, with and without autocast, and
prints the recovery terms of both.  If the hinge is only satisfied under fp16, then the training
signal that was supposed to guarantee repair does not describe the fp32 model that gets evaluated.

Usage::

    python tools/fp16_recovery_check.py --batches 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codetrack.utils.runtime import CPU_THREAD_SETTINGS  # noqa: E402,F401

import torch  # noqa: E402

from codetrack.utils.config import load_config  # noqa: E402

KEYS = ("gain", "gain_active_fraction", "e_ratio_p95", "correct", "preserve", "loss")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/lasher_vitb_corrupt.yaml")
    parser.add_argument("--checkpoint", default="outputs/ab_full/final.pth")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    from codetrack.engine.trainer import Trainer  # noqa: E402

    trainer = Trainer(cfg, output_dir=Path("outputs/fp16_recovery_check"))
    if args.checkpoint and Path(args.checkpoint).exists():
        from codetrack.utils.checkpoint import load_checkpoint
        load_checkpoint(args.checkpoint, trainer.model, map_location=str(trainer.device))
        trainer.logger.info("loaded %s", args.checkpoint)
    model = trainer.model.train()
    if trainer.loader is None:
        # the loader is built by train(); build it here so the check can run standalone
        trainer.build_loader()
    corruption = trainer._corruption_cfg()

    print(f"{'batch':<6} {'mode':<8} " + " ".join(f"{key:>20}" for key in KEYS))
    totals = {mode: {key: 0.0 for key in KEYS} for mode in ("fp32", "fp16", "eval")}
    seen = 0
    for batch in trainer.loader:
        inputs = trainer._to_device(batch)
        target = batch["gt_box"].to(trainer.device, non_blocking=True)
        for mode in ("fp32", "fp16", "eval"):
            trainer.optimizer.zero_grad(set_to_none=True)
            if mode == "fp16":
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=trainer.amp):
                    outputs = model(*inputs, corruption=corruption, clean_teacher=True)
            elif mode == "eval":
                # the evaluation path runs the model in eval() mode and fp32; if the hinge is
                # only satisfied in train() mode, the training signal does not describe the
                # model that is being evaluated
                model.eval()
                with torch.no_grad():
                    outputs = model(*inputs, corruption=corruption, clean_teacher=True)
                model.train()
            else:
                outputs = model(*inputs, corruption=corruption, clean_teacher=True)
            losses = trainer.loss_fn(outputs, target)
            row = [float(losses[key]) for key in KEYS]
            for key, value in zip(KEYS, row):
                totals[mode][key] += value
            print(f"{seen:<6} {mode:<8} " + " ".join(f"{value:>20.4f}" for value in row))
        seen += 1
        if seen >= args.batches:
            break

    print(f"\nmeans over {seen} batches")
    for mode in ("fp32", "fp16", "eval"):
        means = {key: totals[mode][key] / max(seen, 1) for key in KEYS}
        print(f"  {mode:<5} " + " ".join(f"{key}={means[key]:.4f}" for key in KEYS))
    print("\n`gain` is the hinge value and `gain_active_fraction` the share of samples where it "
          "binds: 0.0000 / 0.00 means 'the branch reduced the damaged-token error by more than "
          "10 %', which is what training logs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
