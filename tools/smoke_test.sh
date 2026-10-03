#!/usr/bin/env bash
# Minimal end-to-end sanity run: a handful of iterations on a tiny subset.
set -euo pipefail

OUT="outputs/smoke"
mkdir -p "${OUT}"

PY="${PYTHON_BIN:-python}"
CONFIG="${1:-configs/experiment/lasher_vitb_minimal.yaml}"

echo "[smoke] config=${CONFIG}  out=${OUT}"
"${PY}" -m codetrack.cli.train \
  --config "${CONFIG}" \
  --out-dir "${OUT}" \
  --max-iters 30 \
  --subset 96 2>&1 | tee "${OUT}/smoke.log"

echo "[smoke] done. Check ${OUT}/smoke.log"
