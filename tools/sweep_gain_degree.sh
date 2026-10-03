# Two questions, one sweep.
#
# (1) L_gain: compare its current setting with a fixed hinge margin after recovery metrics
#     and training diagnostics have been checked on the same corruption sample.
#
# (2) Degree sweep uses a nonlocal support so changing degree does not also change the
#     candidate window. Balanced supports require at least 32 edges/check for M=16, N=256,
#     and min variable degree 2.
#
# Block corruption throughout with obs_energy OFF. L_gain groups use local window 5;
# degree groups override it to 0 so degree and locality do not move together.
set -u
PY=/root/autodl-tmp/lab/envs/gola/bin/python
CFG=configs/experiment/lasher_vitb_corrupt.yaml
COMMON="--max-iters 1500 --override train.num_workers=2 --override train.log_every=250 \
        --override model.syndrome_use_obs_energy=false --override corruption.token=[tok_block_erase] \
        --override model.h_locality_window=5 --override model.h_balance_degrees=true \
        --override model.h_links_per_check=32 --override model.h_free_edge_frac=0.25"

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
for d in 32 40 48 64; do
  run "deg$d" --override model.h_locality_window=0 \
      --override model.h_links_per_check=$d
done

sleep 5
tmux ls | grep -c gd_
