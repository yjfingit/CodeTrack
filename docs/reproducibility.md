# Reproducibility

## Environment

| Item | Value |
|---|---|
| Python | 3.10+ |
| PyTorch | >= 2.1 |
| CUDA | 12.x |
| Reference GPU | 1x RTX 4090 D (24 GB) |

`outputs/<exp_name>/resolved_config.yaml` and `outputs/<exp_name>/env.txt`
(`pip freeze` + `nvidia-smi`) are written on every run.

## Seeding

Seeds are set for `random`, `numpy`, `torch` and `torch.cuda`, and `torch.use_deterministic_algorithms`
is enabled where supported. The seed is stored in the run config; corruption streams reuse it.

## Training protocol

| Setting | Value |
|---|---|
| Epochs | 10 |
| Samples per epoch | 131072 |
| Batch size | 8 (ViT-B/16 on 24 GB) |
| Optimizer | AdamW (TODO: lr / wd) |
| Train split | LasHeR train only |
| Test split | never touched during development |

## Making a claim

Every number in `docs/results.md` must be reproducible by a single command:

```bash
bash scripts/train.sh configs/experiment/<exp>.yaml
bash scripts/eval.sh   configs/experiment/<exp>.yaml outputs/<exp>/best.pth
```

Record in the results table: config path, commit hash, seed, GPU, and log path.

## Known sources of variance

- Non-deterministic CUDA kernels in attention and `scatter` ops.
- Dataloader worker count affecting sample ordering when `shuffle` uses a fixed seed but
  variable worker pre-fetch.
- Corruption applied on GPU vs. CPU may change RNG consumption order.

Mitigation: fix `num_workers`, seed both the loader and the corruption RNG separately.
