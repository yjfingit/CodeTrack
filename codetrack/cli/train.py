"""``codetrack-train`` entry point."""

from __future__ import annotations

import argparse

import torch

from ..engine.trainer import Trainer
from ..utils.config import load_config
from ..utils.provenance import run_provenance


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train CodeTrack (RGB-T tracking)")
    p.add_argument("--config", required=True, help="experiment yaml")
    p.add_argument("--out-dir", default="outputs/run")
    p.add_argument("--pretrained", default="checkpoints/pretrained/OSTrack_ep0300.pth.tar",
                   help="OSTrack checkpoint used to initialise the backbone and head")
    p.add_argument("--max-iters", type=int, default=None, help="stop after N iterations")
    p.add_argument("--subset", type=int, default=None, help="samples per epoch (smoke runs)")
    p.add_argument("--device", default=None)
    p.add_argument("--override", action="append", default=[], help="config override key.sub=value")
    return p


def main() -> None:
    import json
    from pathlib import Path

    args = build_parser().parse_args()
    cfg = load_config(args.config, args.override)

    if args.pretrained:
        cfg.setdefault("model", {})["pretrained"] = args.pretrained
    if args.subset:
        cfg.setdefault("data", {})["samples_per_epoch"] = args.subset
    if args.device:
        cfg["device"] = args.device

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    if torch.cuda.is_available():
        (out_dir / "env.txt").write_text(
            f"torch {torch.__version__}\ncuda {torch.version.cuda}\n"
            f"gpu {torch.cuda.get_device_name(0)}\n")

    trainer = Trainer(cfg, output_dir=out_dir)
    # Which code and which run-time switches produced this run.  Written before training starts
    # so an interrupted run is still attributable; --override values are already inside
    # resolved_config.json, but the decoder state mode is a *function* change that the config
    # alone does not make obvious, so it is recorded explicitly (docs/results.md 6.27).
    (out_dir / "run_provenance.json").write_text(json.dumps(
        run_provenance(out_dir.parent, model=trainer.model,
                       extra={"resolved_config": str(out_dir / "resolved_config.json"),
                              "overrides": list(args.override),
                              "max_iters": args.max_iters}),
        indent=2, default=str))
    trainer.train(max_iters=args.max_iters)


if __name__ == "__main__":
    main()
