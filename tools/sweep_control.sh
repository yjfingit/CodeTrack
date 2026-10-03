# Is the locality result real, or is window=16 just small enough to contain a burst?
#
# window=16 gives pearson 0.978 on burst erasure (vs 0.793 with no prior, 0.007 on random
# erase).  Two explanations fit that number equally well right now:
#
#   (a) the locality prior aligns H with visual space, so a spatially contiguous burst
#       becomes a check-level pattern;
#   (b) a 40-token burst fits inside a 16-token window, so each check sees either "mostly
#       wiped" or "mostly fine" -- a much easier target, and nothing to do with geometry.
#
# (b) is a confound of exactly the kind that produced the obs_energy mistake, so it has to be
# excluded before any of these numbers go in the paper.  Three runs:
#
#   burst_w8    window smaller than the burst -> the easy case should still work, but if the
#               gain vanishes once the window cannot cover the burst, the target got easier
#               rather than the prior being useful.
#   burst_w8_loud  same window, severity 0.8 (much bigger perturbation) -> a real structural
#               prior should survive; an amplitude effect should not care either way, so this
#               separates "the signal is spatial" from "the signal is magnitude".
#   burst_w128  window larger than the grid -> degenerates to (almost) no prior, reproducing
#               the w=off baseline and confirming the sweep is monotonic in the prior.
#
# degree is held at 32 and the obs_energy bypass is OFF throughout, per the previous round.
set -u
PY=/root/autodl-tmp/lab/envs/gola/bin/python
CFG=configs/experiment/lasher_vitb_corrupt.yaml
COMMON="--max-iters 1500 --override train.num_workers=2 --override train.log_every=250 \
        --override model.syndrome_use_obs_energy=false --override model.h_links_per_check=32"

run () {  # name token window severity
  name=$1; tok=$2; win=$3; sev=$4
  tmux new-session -d -s "$name" \
    "cd /root/autodl-tmp/lab/projects/CodeTrack && $PY -u -m codetrack.cli.train \
       --config $CFG --out-dir outputs/ctrl_$name $COMMON \
       --override corruption.token=[$tok] --override corruption.ratio=0.2 \
       --override corruption.severity=$sev --override model.h_locality_window=$win \
       2>&1 | tee /root/autodl-tmp/ctrl_$name.log"
  echo "launched $name ($tok, window=$win, severity=$sev)"
}

run burst_w8       tok_burst_erase 8   0.4
run burst_w8_loud  tok_burst_erase 8   0.8
run burst_w128     tok_burst_erase 128 0.4
# the w=16 reference, re-run here so all four share one script and one seed path
run burst_w16      tok_burst_erase 16  0.4

sleep 5
tmux ls | grep -c ctrl_
