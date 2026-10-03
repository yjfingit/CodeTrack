#!/usr/bin/env bash
# Minimal end-to-end sanity run: a handful of iterations on a tiny subset.
set -euo pipefail

OUT="outputs/smoke"
mkdir -p "${OUT}"

echo "[smoke] running 20 iterations on a tiny subset ..."
python -m codetrack.cli.train \
  --config configs/experiment/lasher_vitb.yaml \
  --out-dir "${OUT}" \
  --max-iters 20 \
  --subset 8 2>&1 | tee "${OUT}/smoke.log"

echo "[smoke] done. Check ${OUT}/smoke.log"
