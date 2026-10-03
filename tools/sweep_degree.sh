# Degree sweep with M fixed and support degrees balanced.
#
# "先固定 M=16，只扫 degree={32,40,48,64}；找到拐点以后再扫 M，而且比较 M 时要控制总边数
#  或平均 token degree，避免 confound。"
#
# Why M must NOT move at the same time: raising M at fixed links_per_check changes the total
# edge count and average token check-degree. Match realized support degrees before comparing.
#
# Stage 1 (this script): M=16 fixed and one degree setting changes per group.
set -u
PY=/root/autodl-tmp/lab/envs/gola/bin/python
CFG=configs/experiment/lasher_vitb_corrupt.yaml
COMMON="--max-iters 1500 --override train.num_workers=2 --override train.log_every=250 \
        --override model.h_balance_degrees=true --override model.h_free_edge_frac=0.25 \
        --override model.h_locality_window=0 --override model.syndrome_use_obs_energy=false"

run () {  # name  extra-overrides...
  name=$1; shift
  tmux new-session -d -s "$name" \
    "cd /root/autodl-tmp/lab/projects/CodeTrack && $PY -u -m codetrack.cli.train \
       --config $CFG --out-dir outputs/deg_$name $COMMON \
       --override model.syndrome_use_obs_energy=false $* \
       2>&1 | tee /root/autodl-tmp/deg_$name.log"
  echo "launched $name $*"
}

for d in 32 40 48 64; do
  run "d$d" --override model.h_links_per_check=$d
done

sleep 5
tmux ls | grep -c deg_
