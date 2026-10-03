"""`codetrack-eval` entry point."""


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
    p.add_argument("--root", default=None, help="dataset root override")
    p.add_argument("--device", default=None)
    p.add_argument("--override", action="append", default=[])
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config, args.override)
    if args.device:
        cfg["device"] = args.device

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(cfg, output_dir=out_dir)
    summary = trainer.evaluate(checkpoint=args.checkpoint, subset=args.subset,
                               max_sequences=args.max_sequences, max_frames=args.max_frames,
                               root=args.root)

    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: round(v * 100, 2) for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
