# How does the local candidate window affect a 2-D block corruption?
#
# Earlier exploratory runs reported different syndrome correlations across these settings.
# Re-measure them with degree-matched supports and held-out sequences before interpreting the
# result. Two explanations can produce a high correlation:
#
#   (a) the locality prior aligns H with visual space, so a spatially contiguous block
#       becomes a check-level pattern;
#   (b) a window changes the per-check density labels, changing target difficulty as well.
#
# (b) changes the label difficulty and must be separated from model efficacy. The runs:
#
#   block_w3 / block_w5 / block_w7 vary the local candidate window.
#   block_w5_loud uses a second corruption severity at one geometry.
#   block_w0 is the nonlocal support control.
#
# Degree is held at 32 with matching row and column degrees; obs_energy is OFF throughout.
set -u
PY=/root/autodl-tmp/lab/envs/gola/bin/python
CFG=configs/experiment/lasher_vitb_corrupt.yaml
COMMON="--max-iters 1500 --override train.num_workers=2 --override train.log_every=250 \
        --override model.syndrome_use_obs_energy=false --override model.h_links_per_check=32 \
        --override model.h_balance_degrees=true --override model.h_free_edge_frac=0.25"

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

run block_w0       tok_block_erase 0 0.4
run block_w3       tok_block_erase 3 0.4
run block_w5       tok_block_erase 5 0.4
run block_w5_loud  tok_block_erase 5 0.8
run block_w7       tok_block_erase 7 0.4

sleep 5
tmux ls | grep -c ctrl_
