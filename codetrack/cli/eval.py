"""``codetrack-eval`` entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..engine.trainer import Trainer
from ..utils.config import load_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate CodeTrack on LasHeR")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None, help="CodeTrack checkpoint (optional)")
    p.add_argument("--out-dir", default="outputs/eval")
    p.add_argument("--subset", default="testingset", choices=["testingset", "trainingset"])
    p.add_argument("--max-sequences", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--sequence-list", default=None,
                   help="file with one sequence per line; replaces the subset listing "
                        "(used for the stratified validation split)")
    p.add_argument("--root", default=None, help="dataset root override")
    p.add_argument("--device", default=None)
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--corrupt", action="store_true",
                   help="evaluate under the corruption protocol and report the "
                        "error-correction diagnostics (AUROC / recall / recovery)")
    p.add_argument("--corrupt-identity", action="store_true",
                   help="strict no-op control: diagnostics on, no image or token corruption, "
                        "no RNG draw (unlike --corrupt-ratio 0, which still erases one token "
                        "per frame)")
    p.add_argument("--diagnostics", action="store_true",
                   help="collect the per-frame diagnostics even without corruption, so the "
                        "extra clean-reference forward is present in both conditions")
    p.add_argument("--corrupt-token", default=None)
    p.add_argument("--corrupt-rgb", action="append", default=[],
                   choices=["rgb_lowlight", "rgb_overexp", "rgb_occl"])
    p.add_argument("--corrupt-tir", action="append", default=[],
                   choices=["tir_noise", "tir_crossover"])
    p.add_argument("--corrupt-cross-modal", action="store_true")
    p.add_argument("--corrupt-ratio", type=float, default=0.2)
    p.add_argument("--corrupt-severity", type=float, default=0.4)
    p.add_argument("--corrupt-target", default="both", choices=["both", "rgb", "tir"])
    p.add_argument("--topk", type=int, default=5)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config, args.override)
    if args.device:
        cfg["device"] = args.device

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corruption = None
    if args.corrupt_identity:
        corruption = {"enabled": True, "mode": "identity"}
    elif args.corrupt:
        token_corruptions = [args.corrupt_token] if args.corrupt_token else []
        if not token_corruptions and not args.corrupt_rgb and not args.corrupt_tir \
                and not args.corrupt_cross_modal:
            token_corruptions = ["tok_random_erase"]
        corruption = {
            "enabled": True,
            "token": token_corruptions,
            "rgb": args.corrupt_rgb,
            "tir": args.corrupt_tir,
            "cross_modal": args.corrupt_cross_modal,
            "ratio": args.corrupt_ratio,
            "severity": args.corrupt_severity,
            "target": args.corrupt_target,
        }

    trainer = Trainer(cfg, output_dir=out_dir)
    summary = trainer.evaluate(checkpoint=args.checkpoint, subset=args.subset,
                               max_sequences=args.max_sequences, max_frames=args.max_frames,
                               root=args.root, corruption=corruption, topk=args.topk,
                               sequence_list=args.sequence_list,
                               collect_diagnostics=bool(args.diagnostics))

    summary["invocation"] = {
        "config": args.config,
        "overrides": args.override,
        "checkpoint": args.checkpoint,
        "subset": args.subset,
        "sequence_list": args.sequence_list,
        "max_sequences": args.max_sequences,
        "max_frames": args.max_frames,
        "root": args.root,
        "corruption": corruption,
        "collect_diagnostics": bool(args.diagnostics),
        "topk": args.topk,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    printable = {k: round(v, 4) for k, v in summary.items()
                 if isinstance(v, (float, int))}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
