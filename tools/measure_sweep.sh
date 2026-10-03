#!/usr/bin/env bash
# Measure every checkpoint from sweep_corruption.sh / sweep_degree.sh.
#
# Decision metric (unchanged, and the one the review asked for):
#   softbce_model < softbce_constant   AND   pearson > 0
# reported PER CORRUPTION TYPE, with and without the obs_energy bypass, so the claim
# "the Tanner structure produced the syndrome" can be made on a run that has no label leak.
#
# The measurement corruption MUST match training: the leak is specific to zero-erasure, so
# measuring a feat_noise-trained model under zero-erasure would say nothing about it.
set -u
PY=/root/autodl-tmp/lab/envs/gola/bin/python
cd /root/autodl-tmp/lab/projects/CodeTrack

printf "%-22s %8s %8s %9s %9s %8s %6s\n" \
  "run" "pearson" "spearman" "sbc_model" "sbc_const" "d_std" "beats"

row () {  # dir  token  energy  extra-override
  dir=$1; tok=$2; en=$3; ov=$4
  name=$(basename "$dir")
  [ -f "$dir/final.pth" ] || return
  en_flag=""; [ "$en" = "1" ] && en_flag="--energy"
  out="outputs/fit_${name}.json"
  if ! timeout 300 $PY -u tools/syndrome_fit.py --frames 40 \
        --checkpoint "$dir/final.pth" --out "$out" \
        --token "$tok" $en_flag $ov > /tmp/fit_$name.log 2>&1; then
    echo "$name FAILED -> /tmp/fit_$name.log"; return
  fi
  $PY - "$out" "$name" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
print("%-22s %8.4f %8.4f %9.4f %9.4f %8.4f %6s" % (
    sys.argv[2], d["pearson"], d["spearman"], d["softbce_model"],
    d["softbce_constant_mean"], d["density_std"],
    "YES" if d["softbce_model"] < d["softbce_constant_mean"] - 1e-4 else "no"))
PYEOF
}

# --- corruption sweep: does the syndrome survive without the bypass? -------------
row outputs/sw_erase_noenergy  tok_random_erase 0 ""
row outputs/sw_erase_energy    tok_random_erase 1 ""
row outputs/sw_noise_noenergy  tok_feat_noise    0 ""
row outputs/sw_noise_energy    tok_feat_noise    1 ""
row outputs/sw_burst_noenergy  tok_burst_erase   0 ""
row outputs/sw_burst_energy    tok_burst_erase   1 ""

# --- degree sweep, M fixed at 16, bypass OFF --------------------------------------
for d in 8 12 16 32; do
  row "outputs/deg_d$d" tok_random_erase 0 "--override model.h_links_per_check=$d"
done
row outputs/deg_d32_energy tok_random_erase 1 "--override model.h_links_per_check=32"
