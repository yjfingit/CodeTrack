set -u
PY=/root/autodl-tmp/lab/envs/gola/bin/python
cd /root/autodl-tmp/lab/projects/CodeTrack
printf "%-18s %8s %8s %9s %9s %8s %6s\n" "run" "pearson" "spearman" "sbc_model" "sbc_const" "d_std" "beats"
for w in 0 3 5 7; do
 for tok in block erase; do
  d=outputs/loc_${tok}_w${w}
  [ -f "$d/final.pth" ] || continue
  tk=$([ "$tok" = block ] && echo tok_block_erase || echo tok_random_erase)
  o="outputs/fit_loc_${tok}_w${w}.json"
  timeout 300 $PY -u tools/syndrome_fit.py --frames 40 --checkpoint "$d/final.pth" \
    --out "$o" --token "$tk" --override model.h_locality_window=$w > /tmp/fl_${tok}_$w.log 2>&1 \
    || { echo "${tok}_w${w} FAILED"; continue; }
  $PY -c "
import json,sys
d=json.load(open('$o'))
print('%-18s %8.4f %8.4f %9.4f %9.4f %8.4f %6s' % ('${tok}_w${w}', d['pearson'], d['spearman'],
  d['softbce_model'], d['softbce_constant_mean'], d['density_std'],
  'YES' if d['softbce_model'] < d['softbce_constant_mean']-1e-4 else 'no'))"
 done
done
