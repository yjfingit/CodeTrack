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
PY=/root/autodl-tmp/lab/envs/gola/bin/python
cd /root/autodl-tmp/lab/projects/CodeTrack
SEQ="${SEQ:-20}"
FRAMES="${FRAMES:-80}"

for arm in nodec_clean nodec_corr mlp_corr full; do
  ck="outputs/ab_${arm}/final.pth"
  [ -f "$ck" ] || { echo "$arm: no checkpoint"; continue; }
  ov=""
  case "$arm" in
    nodec_*) ov="--override model.decoder_mode=off" ;;
    mlp_corr) ov="--override model.decoder_mode=mlp" ;;
  esac
  echo "##### $arm"
  timeout 1200 $PY -u -m codetrack.cli.eval --config configs/experiment/lasher_vitb_corrupt.yaml \
      --checkpoint "$ck" $ov --max-sequences "$SEQ" --max-frames "$FRAMES" \
      2>&1 | grep -E "evaluation on|error-correction|skipped|recovery" | head -4
done
