# Contributing to CodeTrack

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
pre-commit install
```

## Before opening a PR

```bash
make lint
make test
make smoke          # 20-iteration sanity run, must pass on CPU or a single GPU
```

## Branch naming

| Prefix | Use |
|---|---|
| `feat/...` | new functionality |
| `fix/...` | bug fix |
| `exp/...` | experiment-only changes (configs, scripts) |
| `docs/...` | documentation only |

## Rules

1. One architecture block per sub-package under `codetrack/models/` — see `docs/architecture.md`.
2. New modules must be gated by a config flag (default off) so baseline results stay reproducible.
3. Never commit weights, datasets, run outputs or anything under `data/`, `checkpoints/`, `outputs/`.
4. Any number published in `docs/results.md` must ship with the config and command that produced it.
5. Keep scripts in `scripts/` thin — real logic belongs in `codetrack/cli/`.

## Reporting issues

Include: commit hash, `configs/` file used, full command, environment (`pip freeze`), and the
complete traceback. For performance issues, attach `nvidia-smi` and the run log.
