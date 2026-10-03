#!/usr/bin/env bash
# Thin wrapper around `codetrack-eval`.
set -euo pipefail

CONFIG="${1:?usage: bash scripts/eval.sh <config.yaml> <checkpoint> [extra args...]}"
CKPT="${2:?usage: bash scripts/eval.sh <config.yaml> <checkpoint> [extra args...]}"
shift 2 || true

EXP_NAME="$(basename "${CONFIG%.yaml}")"
OUT_DIR="outputs/${EXP_NAME}/eval"
mkdir -p "${OUT_DIR}"

echo "[eval] config=${CONFIG}  ckpt=${CKPT}  out=${OUT_DIR}"
python -m codetrack.cli.eval --config "${CONFIG}" --checkpoint "${CKPT}" --out-dir "${OUT_DIR}" "$@" 2>&1 | tee "${OUT_DIR}/eval.log"
