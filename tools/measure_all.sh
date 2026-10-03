#!/usr/bin/env bash
# Measure the control sweep (is window=16 real?) and the gain/degree sweep.
#
# Metric, unchanged: softbce_model < softbce_constant AND pearson > 0, measured with the
# obs_energy bypass OFF and under the corruption the checkpoint was trained on.
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

# ---- control sweep: window 8 / 8+loud / 128 / 16 (all burst) ----
fit outputs/ctrl_burst_w8      burst_w8      tok_burst_erase 8   0.4
fit outputs/ctrl_burst_w8_loud burst_w8_loud tok_burst_erase 8   0.8
fit outputs/ctrl_burst_w128    burst_w128    tok_burst_erase 128 0.4
fit outputs/ctrl_burst_w16     burst_w16     tok_burst_erase 16  0.4

# ---- L_gain ablation (burst, window 16, degree 32) ----
fit outputs/gd_gain_on  gain_on  tok_burst_erase 16 0.4
fit outputs/gd_gain_off gain_off tok_burst_erase 16 0.4

# ---- degree under the locality prior (window 16) ----
for d in 4 8 16 24; do
  fit "outputs/gd_deg$d" "loc_deg$d" tok_burst_erase 16 0.4
done
