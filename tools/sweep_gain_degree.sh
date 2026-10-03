# Two questions, one sweep.
#
# (1) L_gain: is it dead weight?  The new logging says gain_active_fraction = 0.00 and
#     e_ratio_p95 = 0.000, i.e. the hinge is genuinely inactive, not rounded.  The review's
#     instruction was: only delete it after the definitions agree AND an ablation shows no
#     change.  lambda_gain = 0 is that ablation.  Also included: beta 0.9 -> 0.8, because if a
#     larger required margin is wanted the lever is beta, not the weight.
#
# (2) degree, re-swept UNDER the locality prior.  The previous degree sweep was measured with
#     a uniform random support and is void.  With the window the candidate pool is small
#     (cand = window cells), so links_per_check above the window size can no longer be
#     honoured -- that interaction has to be mapped, not assumed.
#
# burst corruption throughout, bypass OFF, window 16.
set -u
PY=/root/autodl-tmp/lab/envs/gola/bin/python
CFG=configs/experiment/lasher_vitb_corrupt.yaml
COMMON="--max-iters 1500 --override train.num_workers=2 --override train.log_every=250 \
        --override model.syndrome_use_obs_energy=false \
        --override corruption.token=[tok_burst_erase] \
        --override model.h_locality_window=16"

run () {  # name extra...
  name=$1; shift
  tmux new-session -d -s "$name" \
    "cd /root/autodl-tmp/lab/projects/CodeTrack && $PY -u -m codetrack.cli.train \
       --config $CFG --out-dir outputs/gd_$name $COMMON $* \
       2>&1 | tee /root/autodl-tmp/gd_$name.log"
  echo "launched $name $*"
}

# ---- (1) does L_gain do anything? ----
run gain_on     --override loss.lambda_gain=1.0 --override model.h_links_per_check=32
run gain_off    --override loss.lambda_gain=0.0 --override model.h_links_per_check=32
run gain_b08    --override loss.gain_beta=0.8 --override model.h_links_per_check=32

# ---- (2) degree under the locality prior (window 16 => cand = 16 cells) ----
for d in 4 8 16 24; do
  run "deg$d" --override model.h_links_per_check=$d
done

sleep 5
tmux ls | grep -c gd_
