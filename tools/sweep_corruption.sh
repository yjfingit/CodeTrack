# Corruption protocol sweep: does the syndrome survive without the energy bypass?
#
# The review's point #1: under zero-erasure the per-check observation energy correlates
# with the corruption density at r = 1.0000, so a syndrome head reading it never has to
# learn any check/variable consistency. Any claim that "the Tanner structure produced the
# syndrome" must therefore be measured WITHOUT that path, and across corruption types whose
# energy statistics differ.
#
# 6 runs = 3 corruption types x {obs_energy off (default), on}.  Same seed, same schedule,
# 1500 iterations each (~200 s), so they are cheap enough to run concurrently.
set -u
PY=/root/autodl-tmp/lab/envs/gola/bin/python
CFG=configs/experiment/lasher_vitb_corrupt.yaml
COMMON="--max-iters 1500 --override train.num_workers=2 --override train.log_every=250 --override train.grad_diag_every=750"

run () {  # name  token  extra-overrides...
  name=$1; tok=$2; shift 2
  tmux new-session -d -s "$name" \
    "cd /root/autodl-tmp/lab/projects/CodeTrack && $PY -u -m codetrack.cli.train \
       --config $CFG --out-dir outputs/sw_$name $COMMON \
       --override corruption.token=[$tok] $* \
       2>&1 | tee /root/autodl-tmp/sw_$name.log"
  echo "launched $name ($tok) $*"
}

# --- zero-erase: the leaking one -------------------------------------------
run erase_noenergy  tok_random_erase  --override model.syndrome_use_obs_energy=false
run erase_energy    tok_random_erase  --override model.syndrome_use_obs_energy=true
# --- additive feature noise: expected energy RISES with the corrupted fraction ---
run noise_noenergy  tok_feat_noise     --override model.syndrome_use_obs_energy=false
run noise_energy    tok_feat_noise     --override model.syndrome_use_obs_energy=true
# --- burst erase: spatially contiguous, still zero-valued ---------------------
run burst_noenergy  tok_burst_erase    --override model.syndrome_use_obs_energy=false
run burst_energy    tok_burst_erase    --override model.syndrome_use_obs_energy=true

sleep 5
tmux ls
