# Locality prior on H's support -- the review's point #4.
#
# "不要把 H 改成手工固定卷积邻域；给可学习 H 加 locality prior 即可。例如每个 check 只在一个
#  局部 candidate window 内学习 top-k support，或者在 edge logits 上加距离 penalty，同时
#  仍让具体边由训练决定。"
#
# Motivation is the burst result: with a uniform random support, a spatially contiguous
# burst is still just an arbitrary fixed-size set to every check, so its spatial continuity
# is washed out before reaching check-space (measured: burst density spread 0.049 < erase
# 0.058). With the bypass off, burst is the one corruption where the syndrome is genuinely
# informative (pearson 0.793), so some alignment already matters -- this makes it a prior
# rather than a happy accident, and lets us ask whether more of it helps.
#
# 6 runs = window {16, 32, 64} x corruption {burst, erase}, bypass OFF, degree 32, M 16.
set -u
PY=/root/autodl-tmp/lab/envs/gola/bin/python
CFG=configs/experiment/lasher_vitb_corrupt.yaml
COMMON="--max-iters 1500 --override train.num_workers=2 --override train.log_every=250 \
        --override model.syndrome_use_obs_energy=false --override model.h_links_per_check=32"

run () {  # name token window
  name=$1; tok=$2; win=$3
  tmux new-session -d -s "$name" \
    "cd /root/autodl-tmp/lab/projects/CodeTrack && $PY -u -m codetrack.cli.train \
       --config $CFG --out-dir outputs/loc_$name $COMMON \
       --override corruption.token=[$tok] --override model.h_locality_window=$win \
       2>&1 | tee /root/autodl-tmp/loc_$name.log"
  echo "launched $name ($tok, window=$win)"
}

for w in 16 32 64; do
  run "burst_w$w" tok_burst_erase  $w
  run "erase_w$w" tok_random_erase $w
done

sleep 5
tmux ls | grep -c loc_
