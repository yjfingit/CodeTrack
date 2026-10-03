# AGENTS.md — instructions for AI coding agents

## Project
CodeTrack: an RGB-T single object tracker built on an error-correction (LDPC / Tanner graph) view of
multimodal representation corruption. See `docs/architecture.md` for the module map and tensor shapes.

## Hard rules for agents working in this repo
1. **Read before write.** Read `docs/architecture.md` and the target module before editing.
2. **Do not refactor the network topology silently.** Module boundaries in `codetrack/models/*`
   mirror the architecture figure; keep one architecture block per sub-package.
3. **Kept equivalence is a contract.** When adding a sub-module, gate it behind a config flag or an
   `enabled: bool = False` switch so the baseline path stays bit-identical.
4. **No dataset mutation.** `data/` is a read-only mount of the lab dataset. Never write there.
5. **Heavy artifacts stay out of git.** Weights -> `checkpoints/`, runs -> `outputs/`.
6. **Tests before claims.** Any numeric claim in docs must be backed by a script under `tools/` or
   an entry in `docs/results.md`.

## Commands
```bash
make lint          # ruff
make test          # pytest
make smoke         # 20-iteration sanity run
bash scripts/train.sh configs/experiment/lasher_vitb.yaml
```

## Conventions
- Python 3.10+, 100-col lines, `ruff` enforced.
- Configs are yaml and support `_base_` inheritance; put experiment-specific knobs in
  `configs/experiment/`, never inline in code.
- Log every run to `outputs/<exp_name>/` with a copy of the resolved config.
