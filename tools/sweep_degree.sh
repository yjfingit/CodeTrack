# Degree sweep with M fixed, per the review's point #3.
#
# "先固定 M=16，只扫 degree={8,12,16,32}；找到拐点以后再扫 M，而且比较 M 时要控制总边数
#  或平均 token degree，避免 confound。"
#
# Why M must NOT move at the same time: raising M at fixed links_per_check raises the total
# edge count, which raises the average token's check-degree and changes support overlap --
# three variables at once.  The earlier M=16 -> 32 run got *worse* (pearson 0.792 -> 0.142)
# and this sweep is what tells us whether that was M itself or the confound.
#
# Stage 1 (this script): M=16 fixed, degree swept.  Stage 2 runs after, with the edge budget
# held constant -- see sweep_degree_stage2.sh.
set -u
PY=/root/autodl-tmp/lab/envs/gola/bin/python
CFG=configs/experiment/lasher_vitb_corrupt.yaml
COMMON="--max-iters 1500 --override train.num_workers=2 --override train.log_every=250"

run () {  # name  extra-overrides...
  name=$1; shift
  tmux new-session -d -s "$name" \
    "cd /root/autodl-tmp/lab/projects/CodeTrack && $PY -u -m codetrack.cli.train \
       --config $CFG --out-dir outputs/deg_$name $COMMON \
       --override model.syndrome_use_obs_energy=false $* \
       2>&1 | tee /root/autodl-tmp/deg_$name.log"
  echo "launched $name $*"
}

for d in 8 12 16 32; do
  run "d$d" --override model.h_links_per_check=$d
done

# reference point: the current default degree, with the bypass ON, to reproduce the 0.792
# that the review is asking about
run "d32_energy" --override model.h_links_per_check=32 \
                 --override model.syndrome_use_obs_energy=true

sleep 5
tmux ls | grep -c deg_
