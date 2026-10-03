# The four controls the review asked for, in the JOINT stage.
#
# Problem with every number so far: `codec_warmup_epochs: 1` and 16384 steps per epoch mean a
# 1500-step run never leaves warmup -- fusion and head stay frozen, so tracking loss sitting at
# 0.55 says nothing about convergence.  So: codec_warmup_epochs = 0 and all four groups get
# the *same* joint budget, same data, same corruption.  Only the named variable differs.
#
#   nodec_clean   no decoder, no corruption   -> baseline tracking ability
#   nodec_corr    no decoder, same corruption -> how much of the gain is plain augmentation
#   mlp_corr      parameter-matched MLP, same corruption -> is it just extra denoising capacity
#   full          complete CodeTrack          -> what the Tanner mechanism adds on top
#
# The comparison that matters is full vs (nodec_corr, mlp_corr): they share the corruption and
# the budget, so a difference cannot be attributed to data augmentation or to capacity.
#
# 6000 steps at ~0.1 s/it is ~10 min per group; all four run concurrently.
set -u
PY=/root/autodl-tmp/lab/envs/gola/bin/python
CFG=configs/experiment/lasher_vitb_corrupt.yaml
STEPS=6000
COMMON="--max-iters $STEPS --override train.num_workers=3 --override train.log_every=200 \
        --override train.codec_warmup_epochs=0 --override train.grad_diag_every=1000"

run () {  # name  extra...
  name=$1; shift
  tmux new-session -d -s "$name" \
    "cd /root/autodl-tmp/lab/projects/CodeTrack && $PY -u -m codetrack.cli.train \
       --config $CFG --out-dir outputs/ab_$name $COMMON $* \
       2>&1 | tee /root/autodl-tmp/ab_$name.log"
  echo "launched $name $*"
}

# decoder_mode="off" is a true pass-through (no message functions, no output LayerNorm), so
# "no decoder" means the same thing in training as in the inference probe.  Zeroing the loss
# weights would NOT: the decoder would still run and still rewrite the tokens.
run nodec_clean --override corruption.enabled=false --override model.decoder_mode=off
run nodec_corr  --override model.decoder_mode=off
run mlp_corr    --override model.decoder_mode=mlp
run full

sleep 5
tmux ls | grep -c ab_
