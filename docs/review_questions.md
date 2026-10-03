# Open questions for an external reviewer

> Question sheet, not a results page. Written after the review round that fixed the
> measurement path; every number below is reproducible from the commands in
> `docs/results.md` section 6 and the artifacts named with it. Assume the numbers are
> correct but small-sample; the *interpretation* and the *next move* are what is being
> asked for.

## 0. Context in one screen

CodeTrack is an RGB-T single-object tracker built on an error-correction (LDPC / Tanner)
reading of multimodal feature corruption. A single learned sparse incidence
`H (16 x 256)` is meant to drive four things at once: check aggregation, the visual
syndrome `s = D(phi(H x), parity)`, the `H^T s` localization vote, and neural BP-style
message passing that repairs tokens with `v <- v + (1 - r) * gate * delta`. Parity is
induced by the same `H` (`A_dyn = H @ w`, `w = softmax_m cos(x_i, U_m)` over template
identity tokens `U`), so check `j` compares what it observes against what the trusted
template says it should observe. Backbone is a frozen OSTrack ViT-B/16 shared by both
modalities (86 M frozen), 26.5 M trainable, corruption is applied *after* the backbone at
the token level (plus an image-level variant), and the correction target is the same
backbone's clean token set.

Repo: `github.com/yjfingit/CodeTrack`, branch `main`. Rules that constrain any proposal
(`AGENTS.md`): do not touch `data/` (read-only mount), do not silently change the network
topology, any new sub-module must sit behind a config flag whose default keeps the
baseline path bit-identical, every numeric claim needs a `tools/` script or a
`docs/results.md` entry, heavy artifacts stay in `outputs/` / `checkpoints/`, keep the
per-process CPU thread caps (`codetrack/utils/runtime.py`).

## 1. What is established (measured, reproducible)

**Measurement path (all fixed this round; previously silent)**

* RGB/TIR pairing: `tools/recovery_probe.py` used `"i" + name[1:]` and dropped whole
  sequences. Now a per-sequence strategy (identical name / prefix swap / modality letter /
  frame number / provably index-aligned positions). `tools/pairing_audit.py`: testingset
  100 % (220 703 frames, old rule left 16 257 unpaired), trainingset 99.75 %, and 0
  disagreements between name-based pairing and the tracker's own positional order.
* Image-level corruption conditions used to raise `KeyError: 'corrupted_rgb'`, which
  `evaluate()` swallowed as "skip sequence" -- every RGB low-light / occlusion / TIR
  crossover condition silently scored **zero** sequences, and their clean reference pass
  reused the degraded frame (AUROC 0.5 by construction). Fixed; the reference pass now uses
  the pristine frame.
* `load_checkpoint` now refuses a stored-vs-live `decoder_mode` mismatch (the `off` arm had
  been evaluated inside a randomly initialised BP decoder: 16 missing tensors). MLP arm: 14
  missing + 8 unexpected. `model.mlp_hidden` is now a config key (default 0 = auto,
  parameter-matched); the stored MLP arm needs `2048` (6.307 M vs 4.928 M for BP).
* `tools/diagnostics.py` at `--frames 10` reproduces the recorded 66.60 -> 66.70 SR /
  0.401 -> 0.330 AUROC bit-identically; at `--frames 30` the same checkpoint scores 50.90.
  The invocation is now stored in `diagnostics.json`.

**Mechanism diagnostics**

* Four-cell probe (`v`, `LN(v)`, `v+D`, `LN(v+D)`) on 240 real frames. An untouched token
  is moved **0.22-0.33 L1** by the output LayerNorm alone. `LN(v+D)` vs `LN(v)` on damaged
  tokens: small gain for `block` (+2.5 %) and `random` (+3.4 %), clearly worse for
  `block_replace` / `feat_noise`.
* Gate probe (`tools/gate_probe.py`, learned / zero / full / oracle). `zero` (full `B x N`)
  reproduces the input exactly in both decoder modes -- the gate plumbing and the identity
  path are correct. For `ab_full` the applied update has cosine **+0.63** with the ideal
  update `clean - v` but **2.2x** its magnitude; forcing the gate open takes the ratio to
  ~10x, the pre-norm error from 0.33 to 0.79, and the post-norm error barely changes
  (0.1919 -> 0.1881). For `block_replace` the direction is orthogonal (cos -0.04).
* Gate information (`tools/gate_information.py`, 48 frames, stable on a 16-frame pilot),
  AUROC against the injected mask: `1 - r` = **1.000**, `H^T s` = 0.853, selection of
  damaged tokens into the graph = 0.812, **learned severity gate within the selected nodes =
  0.392**, **effective gate the decoder applies = 0.147**, syndrome vs per-check density
  rho = 0.764.
* Incidence shuffle: tracking unchanged (66.60 -> 66.70, PR identical) while localization
  collapses (P@5 0.796 -> 0.332 at 5 sequences; 0.760 -> 0.266 at 25 sequences) and the
  density correlation drops 0.807 -> 0.544.

**Validation on 60 stratified testingset sequences (12 per stratum: low illumination,
thermal crossover, total occlusion, partial occlusion, unoccluded), up to 200 frames**

* clean SR (AUC) = **0.340** overall (0.478 unoccluded -> 0.246 low illumination). The same
  checkpoint scores 0.72 on the earlier 20-sequence / 12-frame convenience sample, so the
  two are not comparable. 47 % of sequences have clean SR < 0.2, and the typical failure is
  *not* drift: on `basketballathand` IoU falls 0.78 -> 0.23 within seven frames while the
  target moves ~5 px/frame, while `mandownstair` (slow motion) tracks at 0.80 for all 200.
* Paired clean-vs-corrupt (positive = corruption hurt): RGB block erase 0.2 = **+3.44 pts**
  [0.98, 6.33] p=0.015; RGB 0.4 = **+9.03 pts** [4.92, 13.54] p=0.0001 (relative -21.6 %,
  -27.2 % on the reference-tracking subset); **TIR 0.2 = -4.74 pts** [-9.13, -0.37]
  p=0.040, i.e. corruption *improved* tracking; both-modality 0.2 = -1.36 pts (n.s.).
* Per-stratum drops for TIR 0.2: low illumination +1.69, partial occlusion -1.39, thermal
  crossover -3.77, total occlusion **-9.10**, unoccluded **-11.10**.
* Pre-registered strength screen (20-40 % relative drop, 60-80 % retained SR, interval
  excludes zero): RGB 0.2 = "too weak" (7.9 %), RGB 0.4 = **pass** (21.6 %, retained
  73.4 %), TIR 0.2 = "helps", both 0.2 = n.s.
* Still running or queued: ratio-0.0 control (one erased token per frame, to rule out the
  extra clean-reference forward as the cause of the TIR sign), TIR/both at 0.4, the
  over-parameterised MLP arm on four representative conditions, clean at 20/60 frames
  (horizon calibration), a loss-time distribution probe, and the three-arm summary with the
  paired difference-in-differences.

**Already ruled out, please do not re-suggest**: the pairing bug, the image-level crash,
the decoder-mode mismatch, the MLP width mismatch, the `obs_energy` label leak, the gate
scatter bug, the shuffle-after-load ordering, BP/MLP parameter matching (4.928 M vs
4.929 M), and invocation ambiguity.

## 2. The questions

### Q1. The output space is the thing I distrust most -- what is the correct fix?

`L_correct` and `masked_recovery()` compare the decoder's **post-LayerNorm** output against
**pre-LayerNorm** clean features, and the same function is used for training and for
evaluation. The output LayerNorm is therefore an almost unconstrained transform: it moves an
untouched token by 0.22-0.33 L1, and the gate probe shows the update overshooting the ideal
by 2-10x while the post-norm error stays flat. My reading: the branch is not being asked to
produce a sane residual, it is being asked to produce something that survives the LN.

Candidates I see, and I do not know which is right:

1. keep `v + (1-r) gate delta` as the output and delete the output LayerNorm (the head then
   sees a different input distribution than during training, so this needs a retrain);
2. move the LayerNorm *inside* the update branch (pre-norm: `u = LN(v)`,
   `delta = F(u, ...)`, `v_{l+1} = v_l + alpha (1-r) g delta`), keep the identity path and
   keep the output un-normalised, with a zero-initialised final residual projection;
3. keep the architecture and change the *loss* to compare in the same space
   (`LN(v+D)` vs `LN(clean)`), which at least makes the reported recovery a same-space
   number;
4. predict the residual relative to a normalised input but still emit `v + delta` (the
   "relative residual" reading of the review).

Which is theoretically soundest for a residual denoiser whose teacher is another frozen
network's features, and what ablation would separate "the LN was hiding a scale error" from
"the update direction is genuinely weak"? How do I keep the tracking head's input
distribution stable across those variants so PR/SR stay comparable? Is there a
principled way to constrain the residual magnitude (e.g. a learned scalar scale, a
spectral-norm bound, or normalising by the token's own norm) that a reviewer would accept
rather than read as a tuned hyperparameter?

### Q2. Should the learned severity gate exist at all?

Measured: `(1 - r)` already gates correctly on its own (1 on injected-damage tokens, 0 on
healthy ones, since `r` is trained as `BCE(r, 1 - mask)`). On top of that, the learned
severity gate is *anti*-informative (0.392 within the selected nodes) and the effective
multiplier the decoder applies is 0.147 -- the repair is shrunk exactly where it is needed.
My explanation is that this is the *rational* policy in the current loss geometry: opening
the gate makes the pre-norm error explode while the post-norm loss barely moves, so the
optimiser closes it.

Questions: if the scale/space problem of Q1 is fixed, is a learned severity gate still
justified, or should the correction strength be a *computed* quantity (syndrome magnitude,
`H^T s`, parity residual norm) with no learned MLP? If it stays learned, how should it be
regularised or initialised so it cannot learn "suppress the repair"? And is there a
diagnostic that distinguishes "the gate is compensating for an oversized update" from "the
gate is genuinely learning that some tokens should not be touched" -- without retraining?

### Q3. Is the error-correction framing salvageable, and if so in what exact wording?

Three structural facts I cannot argue away:

1. the decoder aggregates messages per node over the whole `H` and has **no per-edge
   extrinsic exclusion**, so it is not textbook BP and cannot implement a code's iterative
   constraints;
2. "parity" is a soft learned mixture (`A_dyn = H @ w`) rather than an algebraic generator,
   so nothing forces the corrected tokens to lie in a code;
3. corruption is synthetic and applied after a frozen backbone, and the teacher is that same
   backbone's clean features, so the task is feature-space denoising with a known corruption
   model -- not recovery of information that was destroyed in the sensor.

Given those, what is the strongest *honest* claim this architecture can support, and what
would I have to change to make it a real code (discrete/hard parity decisions, iterative
extrinsic BP, a genuine generator matrix, syndrome-based stopping) while staying
differentiable and trainable at this scale? Is it worth doing, or is the honest move to
re-frame the contribution as "learned sparse graph message passing as a robustness
regulariser under structured corruption"? If the latter, what experiment set makes that
claim publishable rather than decorative?

### Q4. The supervision is circular -- what should replace it?

`reliability_rgb` is trained with `BCE(r, 1 - mask)`, `locator_scattered` with
`BCE(locator, mask)`, and the syndrome with the per-check corruption density derived from the
same mask. Hence `1 - r` at AUROC 1.000 is label memorisation, not a capability, and all the
detection/localization numbers share that property. At deployment there is no mask, so it is
unclear what these heads estimate.

What is the best mask-independent supervision for (a) reliability, (b) the syndrome, (c) the
localization vote? Candidates I can think of: temporal consistency (the tracker's own
previous frame / trusted memory), template-to-search consistency, cross-modal mutual
prediction (predict TIR tokens from RGB tokens and vice versa and use the residual),
feature-space reconstruction from the *unmasked* context only, or a contrastive objective
over augmented views. Which of these is both implementable at this scale and measurable in a
way that a reviewer will not call circular? And what is the right *evaluation* for the
correction branch once the mask is not used as a label -- can I even measure "error
corrected" without a mask, or is the honest protocol to hold out corruption *types*
(train on random erasure, test on block replacement / image-level degradation)?

### Q5. TIR corruption improving tracking: mechanism, or artifact?

20 % TIR token block erasure *raises* long-horizon SR by 4.74 points overall (p=0.040) and by
9-11 points in the total-occlusion and unoccluded strata, while RGB-only corruption degrades
it. The improvement is concentrated in sequences whose clean SR is near zero. My leading
hypothesis is that the TIR branch is currently net-harmful (thermal crossover / low-contrast
TIR confusing the fusion), and that erasing TIR tokens effectively down-weights it. Rivals:
(1) an artifact of the extra clean-reference forward in corrupted runs -- a ratio-0.0 control
is running; (2) chaotic trajectory sensitivity: the tracker loses fast targets within ~10
frames, so a small early difference changes everything downstream; (3) the reliability gate
reacting to the corrupted TIR tokens and changing the RGB repair.

What is the minimal, decisive experiment set to separate these? I have TIR 0.4 queued, a
ratio-0 control running, and a matching `ab_nodec_corr` matrix (same corruption, no decoder)
in flight. Should I add a TIR-only arm, a modality-dropout-at-inference probe (feed zeroed
TIR tokens with no mask at all), or a cross-modal alignment diagnostic? If the effect is
real, does it invalidate the "repair the damaged modality with the healthy one" premise, and
how should the paper report it?

### Q6. What should the primary metric be, given that half the sequences are lost from the start?

On the stratified 60-sequence split, clean SR = 0.34 and 47 % of sequences are below 0.2 SR;
the failure mode is "loses a fast target within ~10 frames", not slow drift. SR (AUC) is
therefore bimodal and insensitive to what happens after a sequence is lost, and the paired
drops are diluted. My current protocol reports three views: the full set (primary),
the reference-tracking subset (clean SR > 0.2), and per-stratum. `tools/horizon_probe.py`
measures the loss-time distribution.

Questions: is the reference-tracking subset a legitimate analysis or is it conditioning on
the outcome? Would a loss-time / area-under-IoU-until-loss metric be more sensitive and
still defensible? Should the horizon be shortened (e.g. 60 frames) so the clean arm is
actually tracking, or does that re-introduce the "12 frames is not long-horizon" criticism?
Should I instead use a stronger/cleaner base tracker so that the clean baseline is not
saturated at the floor -- and if so, how do I keep the comparison fair when the correction
branch is attached to it? Finally: how should this be pre-registered so that "we report the
whole curve" cannot be read as picking the level where the effect appears?

### Q7. Which control is the right one, and how much power do I actually have?

The parameter-matched MLP (4.929 M vs BP 4.928 M) answers "is it the capacity", but the
review also asked for a **spatially local mixer** (local pooling/conv + MLP with matched
normalisation, gating and roughly matched compute), because "graph beats per-token MLP" may
only show that spatial information helps. The MLP arm I have (`outputs/ab_mlp_corr`) is
*over*-parameterised (6.307 M, trained with the old auto-width formula) and its checkpoint
predates the current inference code.

Which control should be primary? Should I train: (a) matched MLP, (b) local conv/pool mixer,
(c) both, (d) also a "graph but no parity/syndrome" arm to separate message passing from
check-guided message passing? With 60 sequences, paired bootstrap over sequences and Holm
across conditions, what effect size on the difference-in-differences is detectable at 80 %
power, and how many seeds would the "0.55 PR points" style claim need? Is there a defensible
way to reduce the experiment count (e.g. one pre-registered primary condition, the rest
secondary/exploratory)?

### Q8. What is the minimum credible retraining programme?

All three ablation arms were trained 20:10-20:23 while the last edits to
`codetrack/models/decoder/bp.py` (21:48) and `codetrack/models/codetrack.py` (21:40) landed
afterwards; the MLP width mismatch is the visible symptom. So the numbers I have describe
these weights under the current inference code, not the current architecture. Training one
arm takes ~15 minutes on this box (1 x RTX 4090) when the GPU is free.

What is the minimum set that makes a claim credible: which arms, how many seeds, which
corruption level(s), and which horizon? Should the retrain include the Q1 space fix and the
Q4 supervision change *together* (confounded but cheap) or in two stages (cleaner
attribution, double the compute)? Given that the mechanism diagnostics already say the
current formulation cannot work (gate anti-informative, update overshooting, LN absorbing
the scale), is retraining the *current* architecture at all informative, or should the next
training run only start after Q1/Q4 are settled?

### Q9. Do the geometry claims still mean anything?

The geometry work (locality window on `H`, degree balancing, min column degree 2, realized
degree audits via `tools/locality_geometry.py`) was motivated by "which tokens a check
watches determines what the syndrome can localize". Measured reality: shuffling the incidence
leaves *tracking* untouched and only moves the localization/density readouts; and the
per-check density correlation is already 0.76-0.84. So geometry currently affects the
auxiliary diagnostics, not the tracker.

Should the geometry chapter be (a) kept as a structural property with honest wording,
(b) tied to a tracking metric by making the correction actually matter first, or
(c) dropped? If kept, what is the one-variable-at-a-time sweep that would show a *tracking*
effect, given that `H` changes currently do not move tracking at all?

## 3. What I would like back

1. A ranked set of actions (what to do first, what not to do at all), with the reasoning
   that would let me defend each choice in the paper.
2. For Q1/Q2 specifically, a concrete module-level design (equations plus the
   initialisation/normalisation details) and the ablation table that would separate the
   competing explanations.
3. For Q4, a mask-independent supervision and evaluation protocol that a reviewer would
   accept as non-circular.
4. For Q5, the decisive experiment (or the argument that no experiment on this checkpoint can
   decide it).
5. For Q6/Q7, the statistics: primary metric, primary contrast, multiplicity policy, and a
   power calculation for the retrain.
6. Where you think I am wrong, especially if one of my "established" readings above is a
   misinterpretation of my own measurement.

## 4. Where things live

* Tools added/rewritten this round: `tools/recovery_probe.py` (pairing + four cells),
  `tools/pairing_audit.py`, `tools/gate_probe.py`, `tools/gate_information.py`,
  `tools/horizon_probe.py`, `tools/select_validation_sequences.py`,
  `tools/validate_sequences.sh`, `tools/summarize_paired_conditions.py`,
  `tools/diagnostics.py` (invocation recorded), `tools/eval_corruption_curve.sh`,
  `tools/summarize_corruption_curve.py`.
* Architecture description with the review history: `docs/architecture.md` sections 11-13.
* Protocol and pre-registration: `docs/corruption_protocol.md` sections 3.1 and 6.
* Measurements with commands and artifacts: `docs/results.md` sections 6.1-6.7.
* Tests (61 currently): `tests/unit/test_architecture.py`, `tests/unit/test_recovery_probe.py`,
  `tests/unit/test_evaluation_paths.py`.
