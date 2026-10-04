#!/usr/bin/env bash
# Evaluate one frozen checkpoint on a fixed sequence list under a condition matrix.
#
# Sequences are the sampling unit; every condition runs the *same* sequence list, so
# `tools/summarize_paired_conditions.py` can pair clean-vs-corrupt per sequence.
#
# Usage:
#   CHECKPOINT=outputs/ab_full/final.pth \
#   SEQ_LIST=outputs/validation_split_v1/sequences.txt \
#   OUT_DIR=outputs/validation_ab_full_v1 \
#   TOKEN_LABEL=x_full PARALLEL=5 \
#   bash tools/validate_sequences.sh
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
CONFIG="${CONFIG:-configs/experiment/lasher_vitb_corrupt.yaml}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the frozen checkpoint to evaluate}"
SEQ_LIST="${SEQ_LIST:?set SEQ_LIST to a sequence list file}"
OUT_DIR="${OUT_DIR:?set OUT_DIR for the condition directories}"
FRAMES="${FRAMES:-200}"
PARALLEL="${PARALLEL:-5}"
TOKEN_LABEL="${TOKEN_LABEL:-$(basename "$(dirname "$CHECKPOINT")")}"
# Model-config overrides that the checkpoint requires, e.g.
#   OVERRIDES="model.decoder_mode=off"        for the no-decoder arm
#   OVERRIDES="model.decoder_mode=mlp model.mlp_hidden=2048"  for the old MLP arm
# Without them ``load_checkpoint`` now refuses a decoder-mode mismatch instead of leaving
# that arm's parameters at their random initialisation.
OVERRIDES="${OVERRIDES:-}"

CONDITIONS="${CONDITIONS:-clean tok_block_rgb_02 tok_block_tir_02 tok_block_both_02 tok_block_rgb_04 tok_random_rgb_02 rgb_lowlight_04 rgb_occl_04 tir_crossover_04}"

override_flags() {
  local flags=""
  local key
  for key in $OVERRIDES; do
    flags="$flags --override $key"
  done
  echo "$flags"
}

condition_flags() {
  case "$1" in
    clean) echo "" ;;
    noop_diag) echo "--corrupt-identity --diagnostics" ;;
    tok_block_rgb_02) echo "--corrupt --corrupt-token tok_block_erase --corrupt-target rgb --corrupt-ratio 0.2 --corrupt-severity 0.4" ;;
    tok_block_tir_02) echo "--corrupt --corrupt-token tok_block_erase --corrupt-target tir --corrupt-ratio 0.2 --corrupt-severity 0.4" ;;
    tok_block_both_02) echo "--corrupt --corrupt-token tok_block_erase --corrupt-target both --corrupt-ratio 0.2 --corrupt-severity 0.4" ;;
    tok_block_rgb_04) echo "--corrupt --corrupt-token tok_block_erase --corrupt-target rgb --corrupt-ratio 0.4 --corrupt-severity 0.4" ;;
    tok_block_tir_04) echo "--corrupt --corrupt-token tok_block_erase --corrupt-target tir --corrupt-ratio 0.4 --corrupt-severity 0.4" ;;
    tok_block_both_04) echo "--corrupt --corrupt-token tok_block_erase --corrupt-target both --corrupt-ratio 0.4 --corrupt-severity 0.4" ;;
    tok_random_rgb_02) echo "--corrupt --corrupt-token tok_random_erase --corrupt-target rgb --corrupt-ratio 0.2 --corrupt-severity 0.4" ;;
    # Non-zeroing family.  Every other token condition erases (sets tokens to zero) and the
    # checkpoint was trained on one of those, so this is the condition on which the shipped
    # decoder is measured to *inflate* the error (recovery_gain -0.88, docs/results.md 6.7.1)
    # rather than no-op.  It is the arm comparison's test of the bounded-step hypothesis.
    tok_noise_rgb_04) echo "--corrupt --corrupt-token tok_feat_noise --corrupt-target rgb --corrupt-ratio 0.4 --corrupt-severity 0.4" ;;
    rgb_lowlight_04) echo "--corrupt --corrupt-rgb rgb_lowlight --corrupt-severity 0.4" ;;
    rgb_occl_04) echo "--corrupt --corrupt-rgb rgb_occl --corrupt-severity 0.4" ;;
    tir_crossover_04) echo "--corrupt --corrupt-tir tir_crossover --corrupt-severity 0.4" ;;
    *) echo "__unknown__" ;;
  esac
}

mkdir -p "$OUT_DIR"
echo "checkpoint=$CHECKPOINT sequences=$(wc -l < "$SEQ_LIST") frames=$FRAMES parallel=$PARALLEL"
echo "overrides=${OVERRIDES:-none}"
echo "conditions=$CONDITIONS"

run_one() {
  local condition="$1"
  local flags
  local overrides
  flags="$(condition_flags "$condition")"
  overrides="$(override_flags)"
  if [[ "$flags" == "__unknown__" ]]; then
    echo "unknown condition: $condition" >&2
    return 1
  fi
  # shellcheck disable=SC2086
  "$PYTHON_BIN" -u -m codetrack.cli.eval \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" --sequence-list "$SEQ_LIST" \
    --max-frames "$FRAMES" --out-dir "$OUT_DIR/$condition" $overrides $flags \
    > "$OUT_DIR/$condition.log" 2>&1
}
export -f run_one 2>/dev/null || true
export PYTHON_BIN CONFIG CHECKPOINT SEQ_LIST OUT_DIR FRAMES

started_at="$(date -Is)"
for condition in $CONDITIONS; do
  while [[ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]]; do
    sleep 3
  done
  echo "[$(date +%H:%M:%S)] start $condition"
  run_one "$condition" &
done
wait
echo "[$(date +%H:%M:%S)] all conditions finished"

"$PYTHON_BIN" - "$OUT_DIR" "$TOKEN_LABEL" "$CHECKPOINT" "$SEQ_LIST" "$FRAMES" \
  "$CONDITIONS" "${OVERRIDES:-}" > "$OUT_DIR/run_manifest.json" <<'PY'
import json
import platform
import subprocess
import sys

out_dir, label, checkpoint, seq_list, frames, conditions, overrides = sys.argv[1:8]
manifest = {
    "label": label,
    "checkpoint": checkpoint,
    "sequence_list": seq_list,
    "max_frames": int(frames),
    "conditions": conditions.split(),
    "config_overrides": overrides.split(),
    "cwd": subprocess.run(["pwd"], capture_output=True, text=True).stdout.strip(),
    "cpu_thread_env": {k: v for k, v in __import__("os").environ.items()
                       if k.endswith("_NUM_THREADS") or k.startswith("CODETRACK_")},
    "python": sys.version,
    "platform": platform.platform(),
}
print(json.dumps(manifest, indent=2))
PY

for condition in $CONDITIONS; do
  metrics="$OUT_DIR/$condition/metrics.json"
  if [[ -f "$metrics" ]]; then
    "$PYTHON_BIN" -c "
import json, sys
data = json.load(open(sys.argv[1]))
print(f\"{sys.argv[2]:<22} n={data.get('n_sequences')} PR={data.get('pr', 0) * 100:.2f} \"
      f\"SR={data.get('sr', 0) * 100:.2f} NPR={data.get('npr', 0) * 100:.2f}\")
" "$metrics" "$condition"
  else
    echo "$condition: no metrics.json (see $OUT_DIR/$condition.log)"
  fi
done

echo "started_at=$started_at finished_at=$(date -Is)"
