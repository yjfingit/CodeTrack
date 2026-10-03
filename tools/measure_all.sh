#!/usr/bin/env bash
# Measure the 2-D block-window control sweep and the gain/degree sweep.
#
# These statistics measure syndrome/density association. They do not by themselves establish
# feature recovery, graph-specific benefit, or improved tracking.
set -u
PY=/root/autodl-tmp/lab/envs/gola/bin/python
cd /root/autodl-tmp/lab/projects/CodeTrack

printf "%-20s %8s %8s %9s %9s %8s %6s\n" \
  "run" "pearson" "spearman" "sbc_model" "sbc_const" "d_std" "beats"

show () {  # json name
  [ -f "$1" ] || { echo "$2 MISSING"; return; }
  $PY - "$1" "$2" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
print("%-20s %8.4f %8.4f %9.4f %9.4f %8.4f %6s" % (
    sys.argv[2], d["pearson"], d["spearman"], d["softbce_model"],
    d["softbce_constant_mean"], d["density_std"],
    "YES" if d["softbce_model"] < d["softbce_constant_mean"] - 1e-4 else "no"))
PYEOF
}

fit () {  # dir name token window severity
  dir=$1; name=$2; tok=$3; win=$4; sev=$5
  [ -f "$dir/final.pth" ] || { echo "$name no checkpoint"; return; }
  o="outputs/fit_${name}.json"
  if ! timeout 300 $PY -u tools/syndrome_fit.py --frames 40 --checkpoint "$dir/final.pth" \
        --out "$o" --token "$tok" --ratio 0.2 --severity "$sev" \
        --override model.h_locality_window=$win > /tmp/f_$name.log 2>&1; then
    echo "$name FAILED -> /tmp/f_$name.log"; return
  fi
  show "$o" "$name"
}

# ---- block-window sweep ----
fit outputs/ctrl_block_w0      block_w0      tok_block_erase 0 0.4
fit outputs/ctrl_block_w3      block_w3      tok_block_erase 3 0.4
fit outputs/ctrl_block_w5      block_w5      tok_block_erase 5 0.4
fit outputs/ctrl_block_w5_loud block_w5_loud tok_block_erase 5 0.8
fit outputs/ctrl_block_w7      block_w7      tok_block_erase 7 0.4

# ---- L_gain ablation (block, window 5, degree 32) ----
fit outputs/gd_gain_on  gain_on  tok_block_erase 5 0.4
fit outputs/gd_gain_off gain_off tok_block_erase 5 0.4

# ---- degree with nonlocal support ----
for d in 32 40 48 64; do
  fit "outputs/gd_deg$d" "deg$d" tok_block_erase 0 0.4
done
