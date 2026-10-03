# Locality prior on H's support -- the review's point #4.
#
# "不要把 H 改成手工固定卷积邻域；给可学习 H 加 locality prior 即可。例如每个 check 只在一个
#  局部 candidate window 内学习 top-k support，或者在 edge logits 上加距离 penalty，同时
#  仍让具体边由训练决定。"
#
# Compare 2-D block erasure with random erasure while keeping realized graph degrees matched.
# The random condition tests the spatial-locality hypothesis without assuming that
# independent damage is intrinsically unrecoverable.
#
# 6 runs = window {16, 32, 64} x corruption {burst, erase}, bypass OFF, degree 32, M 16.
set -u
PY=/root/autodl-tmp/lab/envs/gola/bin/python
CFG=configs/experiment/lasher_vitb_corrupt.yaml
COMMON="--max-iters 1500 --override train.num_workers=2 --override train.log_every=250 \
        --override model.syndrome_use_obs_energy=false --override model.h_links_per_check=32 \
        --override model.h_balance_degrees=true --override model.h_free_edge_frac=0.25"

run () {  # name token window
  name=$1; tok=$2; win=$3
  tmux new-session -d -s "$name" \
    "cd /root/autodl-tmp/lab/projects/CodeTrack && $PY -u -m codetrack.cli.train \
       --config $CFG --out-dir outputs/loc_$name $COMMON \
       --override corruption.token=[$tok] --override model.h_locality_window=$win \
       2>&1 | tee /root/autodl-tmp/loc_$name.log"
  echo "launched $name ($tok, window=$win)"
}

for w in 0 3 5 7; do
  run "block_w$w" tok_block_erase  $w
  run "erase_w$w" tok_random_erase $w
done

sleep 5
tmux ls | grep -c loc_
