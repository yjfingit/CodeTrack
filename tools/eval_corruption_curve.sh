#!/usr/bin/env bash
# Evaluate one frozen checkpoint on clean sequences and a fixed token-corruption curve.
# The clean condition is run without --corrupt because ratio=0 is clamped to one token.
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export CODETRACK_TORCH_THREADS="${CODETRACK_TORCH_THREADS:-1}"
export CODETRACK_TORCH_INTEROP_THREADS="${CODETRACK_TORCH_INTEROP_THREADS:-1}"
export CODETRACK_OPENCV_THREADS="${CODETRACK_OPENCV_THREADS:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/experiment/lasher_vitb.yaml}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the frozen checkpoint to evaluate}"
DECODER_MODE="${DECODER_MODE:-off}"
ROOT="${ROOT:-}"
OUT_DIR="${OUT_DIR:-outputs/corruption_curve}"
SEQ="${SEQ:-50}"
FRAMES="${FRAMES:-80}"
TOKEN="${TOKEN:-tok_block_erase}"
TARGET="${TARGET:-rgb}"
SEVERITY="${SEVERITY:-0.4}"

common=(--config "$CONFIG" --checkpoint "$CHECKPOINT" \
        --override "model.decoder_mode=$DECODER_MODE" \
        --max-sequences "$SEQ" --max-frames "$FRAMES")
if [[ -n "$ROOT" ]]; then
  common+=(--root "$ROOT")
fi

echo "=== clean | checkpoint=$CHECKPOINT | sequences=$SEQ frames=$FRAMES ==="
"$PYTHON_BIN" -u -m codetrack.cli.eval "${common[@]}" \
  --out-dir "$OUT_DIR/clean"

for ratio in 0.1 0.2 0.3 0.4; do
  echo "=== $TOKEN | target=$TARGET ratio=$ratio severity=$SEVERITY ==="
  "$PYTHON_BIN" -u -m codetrack.cli.eval "${common[@]}" \
    --out-dir "$OUT_DIR/${TOKEN}_${TARGET}_${ratio}" \
    --corrupt --corrupt-token "$TOKEN" --corrupt-target "$TARGET" \
    --corrupt-ratio "$ratio" --corrupt-severity "$SEVERITY"
done

echo "Metrics saved under $OUT_DIR; all cells used checkpoint $CHECKPOINT."
"$PYTHON_BIN" tools/summarize_corruption_curve.py --out-dir "$OUT_DIR" \
  --token "$TOKEN" --target "$TARGET"
