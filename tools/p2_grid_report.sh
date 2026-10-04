#!/usr/bin/env bash
# Four-arm P2 report: run the pre-registered analysis recipe of docs/results.md 6.10 in order.
#
#   0. gradient conflict (secondary diagnostic, from the training logs)
#   1. update geometry: applied/oracle, L1 gain vs the zero step, gate permutation
#   2. divergence rate per arm (a validity flag, not a metric)
#   3. the two single-factor contrasts at m = 1 (A vs B for the gate, C vs D for the same
#      parameterisation with and without it) and the primary A vs D composite
#   4. the A-vs-D composite split by condition, because a gain that appears only on the erasing
#      family would not support the bounded-step story (6.7.1)
#   5. the clean level check between arms
#
# It waits on nothing: run it after the arm runs exist.  Usage: bash tools/p2_grid_report.sh
set -u
cd "$(dirname "$0")/.."
# The interpreter: honour PYTHON_BIN, else prefer python3 (this repo's runs use an env python and
# a bare `python` is not always on PATH -- which is how the first invocation failed).
if [ -n "${PYTHON_BIN:-}" ]; then
  PY="$PYTHON_BIN"
elif command -v python3 > /dev/null; then
  PY=python3
else
  PY=python
fi
V=outputs/validation_v1
OUT=outputs/p2_report
mkdir -p "$OUT"

echo "=== 0. gradient conflict (training logs) ==="
"$PY" tools/gradient_conflict_report.py \
    --arm "A=outputs/p2_A.log" --arm "B=outputs/p2_B.log" \
    --arm "C=outputs/p2_C.log" --arm "D=outputs/p2_D.log" \
    --out "$OUT/gradient_conflict.json" | tee "$OUT/gradient_conflict.txt"

echo "=== 1. update geometry (step-size probes) ==="
"$PY" tools/compare_step_probes.py \
    --probe "A=outputs/step_size_probe_p2_A.json" --probe "B=outputs/step_size_probe_p2_B.json" \
    --probe "C=outputs/step_size_probe_p2_C.json" --probe "D=outputs/step_size_probe_p2_D.json" \
    --out "$OUT/step_probe_table.json" | tee "$OUT/step_probe_table.txt"
"$PY" tools/compare_step_probes.py \
    --probe "shipped=outputs/step_size_probe_ab_full.json" \
    --probe "A=outputs/step_size_probe_p2_A.json" \
    --probe "D=outputs/step_size_probe_p2_D.json" \
    --out "$OUT/step_probe_vs_shipped.json" | tee "$OUT/step_probe_vs_shipped.txt"

echo "=== 2. divergence rate per arm (validity flag) ==="
"$PY" tools/divergence_report.py \
    --run "p2_A=$V/p2_A" --run "p2_B=$V/p2_B" --run "p2_C=$V/p2_C" --run "p2_D=$V/p2_D" \
    --out "$OUT/divergence.json" | tee "$OUT/divergence.txt"

echo "=== 3. single-factor contrasts and the primary composite (SR) ==="
"$PY" tools/summarize_paired_conditions.py \
    --run "A=$V/p2_A" --run "B=$V/p2_B" --run "C=$V/p2_C" --run "D=$V/p2_D" \
    --reference clean \
    --interaction A B --interaction C D --interaction A D \
    --out "$OUT/summary_four_arms.json" | tee "$OUT/summary_four_arms.txt"
for pair in "A B" "C D" "A D"; do
  set -- $pair
  "$PY" tools/summarize_paired_conditions.py \
      --run "$1=$V/p2_$1" --run "$2=$V/p2_$2" --reference clean \
      --composite "$1" "$2" --composite-conditions tok_block_rgb_04,tok_noise_rgb_04 \
      --out "$OUT/composite_$1_$2.json" | tee -a "$OUT/summary_four_arms.txt"
done

echo "=== 4. primary composite split by condition (A vs D) ==="
for cond in tok_block_rgb_04 tok_noise_rgb_04; do
  "$PY" tools/summarize_paired_conditions.py \
      --run "A=$V/p2_A" --run "D=$V/p2_D" --reference clean \
      --composite A D --composite-conditions "$cond" \
      --out "$OUT/composite_A_D_$cond.json" | tee -a "$OUT/composite_by_condition.txt"
done

echo "=== 5. clean level check between arms ==="
"$PY" tools/summarize_paired_conditions.py \
    --run "A=$V/p2_A" --run "B=$V/p2_B" --run "C=$V/p2_C" --run "D=$V/p2_D" \
    --reference clean --compare A B --compare C D --compare A D \
    --out "$OUT/level_check.json" | tee "$OUT/level_check.txt"

echo "=== 6. IoU endpoint (robustness check, not a tie-breaker) ==="
for pair in "A D" "C D"; do
  set -- $pair
  "$PY" tools/summarize_paired_conditions.py \
      --run "$1=$V/p2_$1" --run "$2=$V/p2_$2" --reference clean --metric iou_mean \
      --composite "$1" "$2" --composite-conditions tok_block_rgb_04,tok_noise_rgb_04 \
      --out "$OUT/composite_iou_$1_$2.json" | tee -a "$OUT/iou_endpoint.txt"
done

echo "report artifacts in $OUT"
