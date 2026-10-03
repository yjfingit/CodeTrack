#!/usr/bin/env bash
# Evaluate the four ablation arms on real LasHeR tracking metrics.
#
# This is the check the review called load-bearing: every number so far came from 1500-step
# diagnostic runs that never left codec_warmup, so nothing connected the mechanism to a
# tracking result.  These four arms share the corruption, the joint-stage schedule
# (codec_warmup_epochs=0) and the step budget; only the named variable differs, so a
# difference between them cannot be attributed to data augmentation or to capacity.
#
#   nodec_clean  no decoder, no corruption -> baseline tracking ability
#   nodec_corr   no decoder, same corruption -> what the augmentation alone buys
#   mlp_corr     parameter-matched MLP, same corruption -> is it just extra denoising capacity
#   full         complete CodeTrack -> what the Tanner mechanism adds
set -u
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export CODETRACK_TORCH_THREADS="${CODETRACK_TORCH_THREADS:-1}"
export CODETRACK_TORCH_INTEROP_THREADS="${CODETRACK_TORCH_INTEROP_THREADS:-1}"
export CODETRACK_OPENCV_THREADS="${CODETRACK_OPENCV_THREADS:-0}"
PY=/root/autodl-tmp/lab/envs/gola/bin/python
cd /root/autodl-tmp/lab/projects/CodeTrack
SEQ="${SEQ:-20}"
FRAMES="${FRAMES:-80}"
TOKEN="${TOKEN:-tok_block_erase}"
RATIO="${RATIO:-0.2}"
SEVERITY="${SEVERITY:-0.4}"
TARGET="${TARGET:-both}"

for arm in nodec_clean nodec_corr mlp_corr full; do
  ck="outputs/ab_${arm}/final.pth"
  [ -f "$ck" ] || { echo "$arm: no checkpoint"; continue; }
  ov=""
  case "$arm" in
    nodec_*) ov="--override model.decoder_mode=off" ;;
    mlp_corr) ov="--override model.decoder_mode=mlp" ;;
  esac
  for condition in clean corrupt; do
    echo "##### $arm / $condition"
    extra=()
    if [ "$condition" = corrupt ]; then
      extra=(--corrupt --corrupt-token "$TOKEN" --corrupt-ratio "$RATIO"
             --corrupt-severity "$SEVERITY" --corrupt-target "$TARGET")
    fi
    timeout 1200 "$PY" -u -m codetrack.cli.eval \
        --config configs/experiment/lasher_vitb_corrupt.yaml \
        --checkpoint "$ck" $ov --max-sequences "$SEQ" --max-frames "$FRAMES" \
        --out-dir "outputs/eval_${arm}_${condition}" "${extra[@]}"
  done
done
