#!/usr/bin/env bash
# Thin wrapper around `codetrack-train`.
set -euo pipefail

CONFIG="${1:?usage: bash scripts/train.sh <config.yaml> [extra args...]}"
shift || true

EXP_NAME="$(basename "${CONFIG%.yaml}")"
OUT_DIR="outputs/${EXP_NAME}"
mkdir -p "${OUT_DIR}"

echo "[train] config=${CONFIG}  out=${OUT_DIR}"
python -m codetrack.cli.train --config "${CONFIG}" --out-dir "${OUT_DIR}" "$@" 2>&1 | tee "${OUT_DIR}/train.log"
