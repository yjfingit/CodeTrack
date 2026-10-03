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

## Git credentials

Credentials are **never** stored in the repository. On a headless GPU box the standard setup is a
stored HTTPS credential:

```bash
git config --global credential.helper store
# writes ~/.git-credentials (chmod 600), one line:
#   https://<user>:<token>@github.com
```

The token is a GitHub personal access token with `Contents: Read and write` on this repository.

Operational rules:

1. Never commit `.git-credentials`, `.env`, or any file containing a token.
2. `.git/push.log` records every automatic push attempt — check it when a commit silently fails to
   appear on GitHub.
3. Rotate (revoke + reissue) the token if it is ever pasted into a chat, log or ticket.

## Known sources of variance

- Non-deterministic CUDA kernels in attention and `scatter` ops.
- Dataloader worker count affecting sample ordering when `shuffle` uses a fixed seed but
  variable worker pre-fetch.
- Corruption applied on GPU vs. CPU may change RNG consumption order.

Mitigation: fix `num_workers`, seed both the loader and the corruption RNG separately.
