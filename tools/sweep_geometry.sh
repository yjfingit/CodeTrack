# Separating geometry from learning, per the review.
#
# "我建议先保留固定支撑方案，修正命名和二维几何，再做三个对照：固定局部支撑＋均匀权重、
#  相同支撑＋学习权重、度数匹配的非局部支撑＋学习权重。这足以回答几何和学习分别贡献什么，
#  不必立即增加可学习拓扑。"
#
# So, four arms, all with the same realized row/column degree distributions and the same
# 2-D block corruption. Degree balancing is an explicit sweep-only config flag.
#
#   geo_unif_local    local support,   uniform weights   <- geometry only
#   geo_learn_local   local support,   learned weights   <- geometry + learning
#   geo_learn_global  random support,  learned weights   <- learning only (degree matched)
#   geo_learn_local32 local support,   learned weights, window 32 (near-global)
#
# `geo_learn_global` is the important one: it has the same degree and the same learned weights,
# only the placement differs.  If the local arms beat it, the geometry is doing something that
# the weights alone cannot reproduce.
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export CODETRACK_TORCH_THREADS="${CODETRACK_TORCH_THREADS:-1}"
export CODETRACK_TORCH_INTEROP_THREADS="${CODETRACK_TORCH_INTEROP_THREADS:-1}"
export CODETRACK_OPENCV_THREADS="${CODETRACK_OPENCV_THREADS:-0}"
PY=/root/autodl-tmp/lab/envs/gola/bin/python
CFG=configs/experiment/lasher_vitb_corrupt.yaml
STEPS=3000
COMMON="--max-iters $STEPS --override train.num_workers=2 --override train.log_every=200 \
        --override train.codec_warmup_epochs=0 --override model.h_links_per_check=32 \
        --override model.h_balance_degrees=true --override model.h_free_edge_frac=0.25"

# Fail before launching GPU work if the realized support degrees diverge.
$PY tools/locality_geometry.py --degree 32 --free-edge-frac 0.25 \
  --balance-degrees --windows 0,5,32 --audit-only

run () {  # name  extra...
  name=$1; shift
  tmux new-session -d -s "$name" \
    "cd /root/autodl-tmp/lab/projects/CodeTrack && $PY -u -m codetrack.cli.train \
       --config $CFG --out-dir outputs/geo_$name $COMMON $* \
       2>&1 | tee /root/autodl-tmp/geo_$name.log"
  echo "launched $name $*"
}

run unif_local   --override model.h_locality_window=5  --override model.h_weight_init=uniform
run learn_local  --override model.h_locality_window=5  --override model.h_weight_init=learned
run learn_global --override model.h_locality_window=0  --override model.h_weight_init=learned
run learn_local32 --override model.h_locality_window=32 --override model.h_weight_init=learned

sleep 5
tmux ls | grep -c geo_
