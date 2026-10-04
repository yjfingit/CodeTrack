# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Capacity-location experiment**: `model.lora.*` installs a frozen-weight, low-rank adapter on the
  Q and V slices of every attention block (`codetrack/models/backbone/lora.py`), optionally per
  modality. `B` is zero-initialised so step 0 is an exact identity, and the adapter's own init
  draws from a *local* generator so it does not shift the global RNG stream and therefore does not
  change any other module's initial weights. `freeze_backbone()` deliberately keeps the adapters
  trainable. Rank 8, per-modality, 12 blocks = **0.590 M** parameters, and
  `load_ostrack_pretrained` aliases the wrapped projection so the OSTrack checkpoint still loads.
  `configs/experiment/lasher_vitb_lora.yaml` is the experiment arm.
- Optimizer **parameter groups**: adapters (`train.lr_lora`), decayed weights, and norms/biases
  without weight decay. With no adapter installed the result is the previous single-rate behaviour.
- `train.schedule` (`constant` default, or `cosine`) with `train.warmup_steps` / `train.max_steps`,
  off by default so archived constant-rate runs stay comparable.
- `tools/search_containment.py` -- is the target still inside the crop the tracker is looking at
  when it fails, and how much does a perfect (lagged ground-truth) crop schedule recover? Records
  containment, centre offset and window-to-object scale ratio; the last two are needed because a
  4x crop square keeps the target "inside" through a drift of more than one object width.
- `tools/token_oracle_interp.py` -- replaces the corrupted fusion inputs with the clean pass's
  tensors (`z_alpha = (1-alpha)*corrupted + alpha*clean`), i.e. an oracle repair, with a per-sequence
  assertion that `alpha = 0` reproduces the plain corrupted run bit for bit. Reports `--include-taps`
  and `--scope foreground|background` variants.
- `tools/inference_protocol_audit.py` -- measures the two protocol differences from the official
  OSTrack tracker (the crop geometry used to map the box back, and the missing score-map window)
  against the historical behaviour, paired on the same sequences.
- `eval.crop_mapping` and `eval.score_window`: both protocol differences are now switchable, both
  default to the historical behaviour. `crop_geometry` in `codetrack/data/transforms/sample.py`
  exposes the geometry `_crop_square` actually takes.
- `docs/results.md` 6.28 -- the two train-free diagnostics and what they jointly say about where
  the failure is.

### Fixed
- **The box mapping ignored the crop's own clamps.** `_crop_square` clamps the side to
  `MAX_CROP_SIDE_FACTOR` times the longer frame edge and pulls the centre onto the frame, but the
  loop mapped the head's normalised box back with the *requested* centre and side. Whenever either
  clamp fired the prediction was placed wrongly by exactly the clamped-minus-requested offset --
  in the diverged regime the clamp exists for -- and nothing else in the pipeline could notice.
  `eval.crop_mapping="actual"` uses the geometry the model was actually given; the default keeps
  the old mapping so archived numbers stay reproducible.
- **The gradient diagnostic crashed on a `lambda_correct = 0` arm.** The weighted reading added in
  the previous round made the *unweighted* `cos(L_track, L_correct)` optional in the diagnostic
  dict, but the log line still indexed it unconditionally, so a lambda sweep died on the first
  logged step. Both parts of the line are now optional.
- `set_trainable()` (the codec warm-up) now also freezes the adapters, so a warm-up stage cannot
  leave them training while the config claims otherwise.

- **Decoder iteration state is now explicit** (`model.decoder_state_mode`: `recurrent` default,
  `held` optional). `073c223` had silently frozen the branch input for every non-identity output
  mode, so `post_norm`/`mlp`/`spatial` changed *function* from round 2 on while the commit and
  `docs/results.md` described the default path as bit-identical. `recurrent` restores the
  pre-`073c223` loop exactly; `load_checkpoint` now refuses a state-mode mismatch like it already
  did for `decoder_output` and `gate_always_one`. Measured by
  `tools/decoder_state_equivalence.py` (`docs/results.md` 6.27.1).
- **The fixed-crop reference was misaligned by one frame.** The free loop crops frame *f* around
  the box from *f − 1*; passing `reference[f]` showed the replay a crop the free run never saw and
  leaked the reference's answer a frame early. `crop_box_for_frame` now lags the schedule and keeps
  frame 0 annotation-driven. Every fixed-crop result, including the TIR reading of 6.12, is
  provisional until re-run.
- **`power_plan.py` reported the wrong probability as "FWER"**: it counted P(this hypothesis
  rejected) rather than P(any rejection in the family), and drew an independent sample per
  comparison, discarding the correlation between contrasts measured on the same sequences. Now
  `family_wise_rejection_rate` with one shared resample (or an (m × n) effect matrix). 6.13's power
  figures are superseded.
- `clamp_box` now returns *why* a prediction was clamped (scale / centre / non-positive /
  non-finite) instead of one boolean, and the evaluator stores the breakdown per sequence and per
  run. Only the scale reason means the closed loop grew without bound.
- `tools/step_size_probe.py` reports `a_applied/a*`, the actual step-multiplier ratio. The
  "34x overshoot" of 6.10/6.24 was `L1(applied)/L1(oracle)`, an **error** ratio, and the two are
  now named separately.
- `tools/horizon_probe.py` stores `failure_confirmed_frame` (the event is knowable only
  `patience − 1` frames after its start) and an RMST endpoint, and 6.17.1's "permanently" /
  "median failure frame 16" wording is corrected to what a 200-frame window can support.
- `tools/trajectory_replay.py`'s `crop_identical` was `allclose(...) or len(gt) > 0`, i.e. always
  true; it now compares the crops the loop actually recorded, frame by frame.
- `tools/summarize_paired_conditions.py` marks every `--composite` result `in_holm_family: false`
  and says so on stdout; 6.26's "+3.65 points, p = 0.013" came from that path and is exploratory.
- `tools/gradient_conflict_report.py` reports the **weighted** auxiliary gradient
  (`cos(g_track, g_aux)`, `‖g_aux‖/‖g_track‖`, by thirds of the run, opposed-and-loud fraction), not
  only the unweighted `cos(L_track, L_correct)`, which cannot separate "repair is ineffective" from
  "repair is harmful". `LossOutput` now exposes the graph-connected copies of every weighted term.

- `codetrack/utils/provenance.py` and `run_provenance.json` per training run / `eval_provenance` +
  `train_provenance` per evaluation manifest: commit, dirty flag, decoder function switches, active
  parameter count, peak memory. `tools/run_provenance_audit.py` reports which existing numbers can
  be attributed to a commit (first run: 76 runs, 0 comparable -- the schema is new, so the round-2
  numbers must be produced again rather than extended).
- `tools/decoder_state_equivalence.py` -- measures `recurrent` vs `held` vs an independently written
  pre-`073c223` loop, and asserts `recurrent` reproduces the parent bit for bit.
- `docs/results.md` 6.27 -- the round-2 truth-fix record, the provenance audit, and the
  pre-registration for the round-2 experiments (primary contrast, m = 1, 3-point minimum useful
  effect, RMST as a mechanistic secondary, and what counts as exploratory).
- Initial repository skeleton: package layout, configs, docs, scripts, tests.
- Architecture documentation with tensor shapes (`docs/architecture.md`).
- `tools/git-hooks/post-commit` — background auto-push to `origin`, installed by
  `tools/install-git-hooks.sh`; `tools/sync-to-github.sh` as a manual fallback.
- `tools/install-gh.sh` — persistent GitHub CLI install (kept on the data disk) with
  device-code login and `gh auth setup-git` wiring.
- **Full CodeTrack implementation following the architecture figure**
  (`codetrack/models/`): shared OSTrack ViT-B/16 backbone with lossless pretrained
  loading, target codebook encoder, reliability estimator, target candidate selector,
  reliability-adaptive Tanner graph, visual syndrome, error locator, neural
  belief-propagation decoder, FPN fusion and the centre/corner tracking head.
- Data and training stack (`codetrack/data/`, `codetrack/engine/`): LasHeR reader,
  template/search sampling, channel corruption simulator, the four-term objective
  `L_track + 1.0·L_detect + 2.0·L_correct + 0.1·L_identity` with a clean teacher, and a
  trainer with a staged codec warm-up.
- Evaluation with PR / SR / NPR (`codetrack/metrics/`, `codetrack/cli/eval.py`).
- `tools/check_shapes.py` — audits all 21 architecture-figure tensor shapes in one
  forward pass (currently 21/21 OK).
- Experiment configs: `lasher_vitb`, `lasher_vitb_corrupt`, `lasher_vitb_minimal`.
- `codetrack/utils/runtime.py` — per-process CPU thread caps (`OMP`/`MKL`/OpenBLAS/
  PyTorch/OpenCV) shared by training, evaluation and every probe, so parallel runs stay
  measurable.
- `tools/pairing_audit.py` — read-only RGB/TIR pairing audit over both LasHeR subsets.
- `tools/gate_probe.py` — learned / zero / full / oracle severity-gate policies on one
  corruption sample, with the applied-vs-ideal update cosine and norm ratio.
- `tools/select_validation_sequences.py` — seeded, stratified validation split from
  LasHeR's own challenge attributes (`AttriSeqsTxt/`).
- `tools/validate_sequences.sh` — one frozen checkpoint x one condition matrix on a fixed
  sequence list, several conditions in parallel, with a `run_manifest.json` per run.
- `tools/summarize_paired_conditions.py` — sequence-level paired statistics (bootstrap
  interval, paired t, Wilcoxon, Holm-corrected p) and cross-checkpoint comparisons, with a
  secondary view restricted to the sequences the reference arm actually tracks, a per-stratum
  table, and `--screen-run` to evaluate the pre-registered strength screen
  (20-40 % relative drop, 60-80 % retained SR, interval excludes zero) per condition, and a
  `--interaction` difference-in-differences for the cross-arm robustness claim.
- `loss.detect_target` + `loss.detect_deviation_cm` — the detection heads can be supervised
  with the continuous feature deviation from the frozen teacher
  (`q_i = e_i / (e_i + c_m)`) instead of the injected corruption mask.  Default stays `mask`,
  so the shipped path is unchanged; `deviation` is what makes the supervision definable for
  degradation families that have no mask, and it is still teacher distillation rather than real
  degradation ground truth.
- `Trainer.infer_sequence(fixed_boxes=...)` + `crop_box_for_frame()` — replay a reference
  trajectory: every condition is cropped around the reference arm's boxes, so a per-frame
  difference is the model's response instead of the closed loop having diverged earlier.  The
  default `None` keeps the free-running behaviour bit-identical.
- `tools/trajectory_replay.py` — fixed-crop replay **and** path interventions.  With crops
  replayed from the clean arm and masks matched across arms, 20 % TIR token erasure costs 0.0017
  mean IoU (0.0194 in the identical-history window), so the free run's apparent gain is
  closed-loop dynamics; forcing the clean reliability/gates onto corrupted features is the worst
  arm (0.2778 vs 0.2823 clean), i.e. the control path is active rather than inert.  Both the
  tool and its first measurement had a defect -- extra forwards consumed the global torch RNG
  between arms, shifting every later arm's masks -- so `--seed` is now applied before every arm
  and section 6.12 carries the reseeded numbers.
- Endpoint verdict recorded in `docs/results.md` 6.17.1: the mean-IoU endpoint does **not** have a
  smaller paired variance than SR (paired clean-level sd 0.0936 vs 0.0912, MDE 0.0338 vs 0.0330 at
  n = 60), so no endpoint switch buys power -- only a larger split or a smaller family does.  The
  same run gives the descriptive headline: 78.3 % of sequences fail within the horizon, the median
  failure frame is **16**, and only 12.8 % of failures recover.
- `tools/divergence_report.py` — the closed-loop divergence rate per condition and per run, with
  the worst sequence, and an explicit "unknown" for runs recorded before the counter existed (0 %
  and "not measured" must not look alike).  First measurement: the MLP arm clips 223 of 51 220
  frames (0.44 %), including 70 in its `clean` run, and one sequence carries the divergence in
  four of its five conditions.
- `tools/summarize_paired_conditions.py --metric {sr,pr,npr,iou_mean}`: the same paired,
  Holm-aware machinery, stratum view, screen and composite can be run on the lower-variance
  mean-IoU endpoint; all metrics present in a run are written to the JSON, and a sequence without
  the field shrinks `n` rather than raising.
- `tools/summarize_paired_conditions.py --composite A B --composite-conditions ...`: reports one
  pre-registered contrast over several conditions (per-sequence mean drop, then the difference in
  differences), optionally normalised by each sequence's clean SR.  This is what makes the
  headline of `docs/results.md` 6.19 reproducible by a committed command (+4.34 SR points,
  [+1.75, +7.24], p = 0.0034 absolute; +15.78 % of clean, [+6.74, +25.01], p = 0.0013 relative).
- `tools/summarize_paired_conditions.py`: the `--interaction` output had **two** sign problems
  after the estimator was flipped.  The help text and header still described the old convention,
  and -- worse -- the printed `D` column was computed from a leftover local `drop_a - drop_b`
  while the estimator and the JSON used `drop_b - drop_a`, so the printed point estimate
  disagreed in sign with the confidence interval on its own line.  The print now reads the stored
  estimate (single source of truth), the header states `D = drop(B) - drop(A)` with "positive =
  the first arm degrades less", and the columns are named `dropA`/`dropB`/`D=dB-dA`.  Every
  aggregation in `docs/results.md` that had been transcribed from the printed column now carries
  the corrected signs, with the correction stated in place.
- Pre-registered minimum credible experiment (`docs/results.md` 6.17), derived from the measured
  variance instead of assumed: primary condition `rgb_occl_04`, primary contrast the
  decoder-vs-branch-off difference in differences with m = 1, sigma_D = 13.14 SR points, i.e.
  4.75 points detectable at n = 60 but **3.01 at n = 150** (2.5x the current split, feasible on
  the 245-sequence testingset).  Effects below ~2 points are out of reach at any realistic size,
  which is why the honest status is "not measured at a resolvable effect size".
- The evaluation now scores the **reliability head** (`1 - r`) against the token mask
  (`reliability_auroc_rgb/_tir` and the mean), per modality, with a degenerate mask reported as
  NaN rather than as 1.0.  It previously had no number in the evaluation path at all: only
  `tools/gate_information.py` scored it, and that tool cannot run the training corruption family,
  so "is AUROC 1.000 memorisation?" was untestable.  Measured across families, it is a
  *mechanism* shortcut: 1.0000 on three zeroing families (two of them held out) and 0.7440 on a
  held-out non-zeroing one.  The same runs show the *repair* side is worse than the detection
  side: `recovery_gain` -0.02 (a ~2 % no-op) on zeroing families but **-0.88 (1.88x the error)**
  on the non-zeroing one, i.e. the branch is specific to the mechanism it was trained on.
- `evaluate()` now records **mean frame IoU per sequence** (`per_sequence[*].iou_mean`, plus an
  aggregate `iou`) alongside SR/PR/NPR.  SR is a thresholded statistic with a large per-sequence
  variance (MDE ~7.9 points at 60 sequences for the strong conditions, docs/results.md 6.13), and
  a paired robustness claim needs a lower-variance endpoint to be powered at a realistic sample
  size.  Additive: existing consumers read SR as before, and `tools/power_plan.py --metric iou`
  refuses a run that predates the field instead of averaging it as zeros.
- `model.decoder_mode="spatial"` — the P5 primary control: a depthwise 3x3 mixer over the token
  grid plus the parameter-matched MLP, with no parity matrix, no messages, no syndrome and no
  locator.  Parameter-matched to BP at the reference size (4.9287 M vs 4.9277 M, +0.02 %), same
  `(1 - r) * gate` gating and the same two output parameterisations; `bp` stays the default and
  a non-square token grid is rejected rather than silently falling back to the pointwise arm.
- `tools/power_plan.py --simulate` — Monte-Carlo power under the **real Holm step-down**,
  resampling the measured per-sequence difference in differences instead of assuming a
  distribution, plus a family-wise error check at effect 0.  Measured for the
  full-vs-no-decoder contrast at 60 sequences: FWER 0.015-0.069, 80 % power needs 5-6 SR points
  at m = 1 and 6-7 at m = 8, so 1-3 point effects are unclaimable from this design.
- `tools/power_plan.py` — power planning from the measured per-sequence paired drops rather
  than assumed variances: it reports sigma for each arm's drop, the difference-in-differences
  sigma, the MDE at a planned n and family size, and the sequences a given effect would need.
  Measured on the 60-sequence split: MDE 7.85 SR points for RGB block erasure 0.4 and 8.42 for
  image occlusion 0.4 (both effects clear it), 4.91 for ratio 0.2 (the effect does not), and
  7.60 points for the full-vs-no-decoder difference in differences (which is why the robustness
  claim is only powered for large effects at n = 60).
- `loss.detect_target="deviation"` is exercised end to end by a CPU training smoke (6 steps,
  finite losses, checkpoint written), not only by unit tests; the P4 chain records the measured
  `detect_deviation_cm` in `outputs/residual_budget_p4.json` instead of assuming one.
- **First positive architecture signal, recorded in `docs/results.md` 6.19.**  The BP decoder arm
  degrades less than the pointwise MLP arm on all four shared conditions (+3.07 to +5.43 SR
  points, raw p 0.014-0.047), and their mean composite is +4.34 points ([+1.75, +7.31],
  p = 0.0034; +15.8 % relative, p = 0.0013) with 72 % of sequences favouring the BP arm.  The MLP
  arm has 28 % *more* decoder parameters, so capacity does not explain it; it is not a matched
  control either, which is what the spatial arm (P5) is for.  Inside the 36-test family the Holm
  values are 0.355-0.900, which is the concrete argument for pre-registering a single contrast.
- `tools/validate_sequences.sh` gains a `tok_noise_rgb_04` condition (`tok_feat_noise`, ratio 0.4):
  every other token condition erases, and the shipped decoder is a ~2 % no-op on erasing but
  inflates the error by 88 % on the non-zeroing mechanism (docs/results.md 6.7.1).  It is the
  condition where a bounded-step arm has the largest predicted effect, so the P2 grid and the
  spatial arm are evaluated on it alongside `clean` and `tok_block_rgb_04`.
- `tools/compare_step_probes.py` — one table for several `step_size_probe` artifacts:
  `applied/oracle`, the L1 gain versus the zero step, and the within-stratum gate-permutation
  effect.  The shipped decoder measures `applied/oracle` 1.677 with the update *hurting*
  (`-0.107`), versus 1.062 and a marginal help when forced to a single step — the P2 arms are
  judged on those ratios before any tracking comparison.
- `tools/step_size_probe.py` — separates "uniform overshoot" from "selective rejection"
  without retraining: per-token oracle step `a*`, the applied step, the L1 error at zero /
  applied / oracle / best-global step, and a gate permutation *within* (reliability, `||d||`)
  strata so only the gate's value information is destroyed. `--single-step` forces
  `bp_iterations=1` because with two iterations the gate changes `v` after the first step and
  the gate-open vs gate-learned forwards no longer differ by a fixed per-token factor.
- `--corrupt-identity` + `--diagnostics` (and the `noop_diag` condition in
  `tools/validate_sequences.sh`): a **strict no-op** control that keeps the diagnostics pass
  enabled but draws no RNG at all. `--corrupt-ratio 0` is not a no-op -- `corrupt_tokens`
  clamps to one erased token per frame. `evaluate()` gained `collect_diagnostics`, so the
  extra clean-reference forward is no longer conditional on corruption being present.
- **P4 result (positive), `docs/results.md` 6.20.**  With `c_m` measured at 0.1626, the
  deviation-target arm beats the mask-target arm on the pre-registered endpoint: reliability AUROC
  on the held-out non-zeroing family **0.8123 vs 0.7440** (+0.0682, above the >= 0.05 the design
  was stated to resolve), keeps 1.0000 on the zeroing family, and its repair operator turns from a
  no-op into a real improvement on the training-like mechanism (`recovery_gain` -0.0196 ->
  **+0.1070**, error 0.2157 -> 0.1926).  On the held-out mechanism the repair is still negative but
  less so (-0.8754 -> -0.5574).  The gain is not collateral damage: `damage_clean` on untouched
  tokens is *lower* for the deviation arm on both families (0.1055 vs 0.1251 and 0.1051 vs 0.1297),
  so it repairs more and perturbs healthy code less.  Not yet shown: that this reaches the tracker
  (a P4 tracking follow-up is appended to the queue, with its endpoint and decision rule
  pre-registered in 6.20 before the run existed: `full` vs `p4` on `tok_noise_rgb_04` at m = 1,
  powered only for a large effect -- MDE about 5-6.5 SR points at n = 60 -- so a null means "no
  large transfer", not "no transfer").
- **Process failure and its fix (`docs/results.md` 6.23).**  The P2 evaluation stage passed no
  runtime overrides, so arms B/C/D were being evaluated with default decoder settings: B without
  `gate_always_one=true` (applying *untrained* gate weights) and C/D without
  `decoder_output=identity_residual` (the load silently dropped `branch_norm`/`residual_out` and
  evaluated a different network).  The runs were stopped after arm A finished (A needs no
  overrides, so 6.22 is valid) and the remaining work moved to a corrected queue.
  `load_checkpoint` now refuses a mismatch in `decoder_output` and `gate_always_one` as well, with
  a unit test, so the mistake fails loudly next time.  Verified against the real checkpoints:
  `p2_A` loads into the default model, `p2_B`/`p2_C`/`p2_D` are refused with the override to pass
  named in the message.
- `docs/results.md` cross-references: every design/pre-registration section now points at its
  result section (6.10 -> 6.24, 6.14 -> 6.25, 6.18 -> 6.20, 6.22 -> 6.24), and 6.19 is explicitly
  read together with 6.25 so that "the BP decoder beats a pointwise MLP" and "a local spatial mixer
  matches the BP decoder" cannot be mistaken for a contradiction.
- `docs/results.md` 6.0 (the claims-status table) now carries the P2 and P4 outcomes: the
  identity-residual geometry fix, the robustness null, arm C's clean result, the divergence cost,
  the gate's level-versus-drop role, the deviation-target result and the spatial control's geometry.
- **P4 tracking follow-up (`docs/results.md` 6.26): the better detector and repairer tracks worse.**
  The deviation-trained arm loses 15.22 SR points on `tok_block_rgb_04` where the mask-trained arm
  loses 9.03 -- a difference in differences of **+6.19 points** (CI [1.03, 11.76], p = 0.028) -- from
  comparable clean levels (0.329 vs 0.339).  Together with 6.24 this is the session's sharpest
  pattern: two independent improvements to the token-level correction (geometry fixed; detection and
  repair improved by the loss) and *neither* produced the expected tracking gain, one of them moved
  it backwards.  Whatever limits this tracker is not the quality of the per-token correction.
- **P4 primary endpoint, and the pre-registration gap closed.**  `ab_full`'s missing
  `tok_noise_rgb_04` reference run was executed (6 minutes) and installed.  Primary endpoint:
  **+1.11 points, CI [-0.57, 2.97], p = 0.227** -- no confirmed difference on the pre-registered
  condition.  Secondary `tok_block_rgb_04`: +6.19 points (p = 0.028).  Composite over both:
  **+3.65 points, CI [1.01, 6.48], p = 0.013**, i.e. the deviation-trained arm is **significantly
  worse in aggregate** despite its better detector and repairer.  The gap is recorded in 6.26 as a
  process lesson: every arm must have every condition the protocol names, and a missing cell must be
  reported rather than silently dropped from a comparison.
- **Clamp audit result (`docs/results.md` 6.15/6.16): the published token-condition numbers
  stand.**  Re-running the three strongest token conditions with the crop bound in place changes SR
  by **+0.53 / +0.01 / -0.85** points (`tok_block_rgb_04` / `tok_block_tir_04` /
  `tok_block_both_02`) -- all within +/-1 point and far below the 4-8 point MDE -- so the unclamped
  diverged crops did not materially contaminate section 6.5.  The fix's value is that a no-result
  failure became a measurement and that divergence is now reportable (3.44 % of frames on
  `tok_block_rgb_04`, invisible before).
- **P5 spatial control result (`docs/results.md` 6.25): ordinary local context accounts for almost
  all of it.**  The parameter-matched spatial mixer (no parity, no messages, no syndrome, no
  locator) reaches 0.353 clean SR and 0.252 on `tok_block_rgb_04`, against the shipped BP arm's
  0.339 and 0.249, and is **statistically indistinguishable from the best BP arm on the robustness
  composite** (+0.10 points, CI [-2.50, +2.55], p = 0.94); on the level the BP arm leads by 2.6-3.9
  points, with only one of three conditions nominally significant (p = 0.044).  The one arm ahead of
  the spatial control is C, and its advantage comes from the output parameterisation and the learned
  gate rather than from the graph (A and D sit 6.5-7.6 points *below* the spatial mixer).  The
  honest next experiment is stated in the section: spatial mixer + `identity_residual` + gate, which
  would separate "the graph contributes" from "the output parameterisation contributes".
- The P5 spatial-mixer control's geometry is recorded in `docs/results.md` 6.10: with the default
  `post_norm` output its update overshoots 2.80x and hurts the token error by 0.30 L1 (`gain vs
  zero` -0.3006) -- between the BP `post_norm` arms (34-35x, -5.8) and the identity-residual arms
  (1.00-1.02x, *positive* gain) -- and its learned gate sits at a median of 0.992.
- The spatial arm's step probe initially failed with the `decoder_mode` load guard (it had been
  queued without `--override model.decoder_mode=spatial`, unlike its evaluations); re-run with the
  override.  That guard is the P0-era one and it is doing its job.
- **P2 grid outcome (`docs/results.md` 6.24).**  Geometry first: the identity-residual arms apply
  1.015/1.004x the ideal step and *reduce* the damaged-token error (+0.0145/+0.0151), the
  `post_norm` arms overshoot 34-35x and raise it sixfold.  But the pre-registered primary contrast
  (A vs D composite over the two token conditions) is **-0.80 points** (CI [-5.10, +3.37],
  p = 0.72), so prediction 1 of 6.22 is not confirmed -- the token-level repair does not propagate
  into robustness at this sample size, and the interval bounds any such effect at ~5 points.  Where
  the arms *do* differ is clean tracking, and there the gate matters: C (`identity_residual` with
  the learned gate) reaches **0.385 clean SR** against A 0.288, B 0.244, D 0.277 and 0.339 shipped
  -- the best of the four by a wide margin -- while the same parameterisation with the gate forced
  to 1 (D) is no better than the `post_norm` arm.  Prediction 3 is therefore half-falsified: the
  gate is inert for the corruption response but not for where training lands.  "The gate is
  decoration" was too strong and is corrected here.  All three single-factor composites are null (A vs B +0.89, C vs D +0.84, A vs D -0.80 points,
  each bounding any effect at ~4-5 SR points), and the gate's role is now precise: it changes the
  *level* (A - B = +4.40 clean points, p = 0.040; C - D = +10.76, p = 0.0002) but not the drop, so it
  is a training-time conditioner rather than a damage detector.  Two additions: the null repeats on
  the mean-IoU endpoint (-0.87 points, CI [-5.31, +3.46], p = 0.70), and the identity-residual arms are
  the ones whose closed loop diverges more (C 1.67 %, D 1.54 % of frames clamped against A 0.14 %
  and B 0 %), with one sequence (`shinycarcoming2`) the worst case in three arm/condition pairs --
  the parameterisation is not free, and the four-arm ranking is not one-dimensional (B is the most
  stable arm and the worst tracker).
- **P2 arm C is the best tracker on clean and on the non-zeroing condition (`docs/results.md`
  6.22).**  Clean SR **0.385** against 0.339 shipped, 0.346 MLP control, 0.288 A and 0.244 B;
  `tok_noise_rgb_04` **0.402** against A 0.314 and B 0.252.  Paired over 60 sequences C beats A by
  9.71 points on clean (p = 0.0005) and 8.86 on the noise condition (p = 0.0005), and is +4.6
  points above the shipped checkpoint -- the direction the 6.10 geometry predicted, and the first
  tracking evidence that the identity-residual parameterisation is an improvement in its own right.
- **Arm C's `tok_block_rgb_04` run aborted** (`CUDA error: unspecified launch failure`) and the
  evaluator skipped 51 of 60 sequences, so its reported SR sits on 9 sequences; the condition is
  re-run with `PARALLEL=1` (file-state wait, not a process-name wait).  Recorded in 6.22, including
  the admission that the first reading of the table did not check `n_skipped`.
- **P2 A vs B: the learned gate is decoration, in tracking as well as in geometry.**  With the
  identical `post_norm` parameterisation, forcing the gate to 1 changes the two-condition composite
  difference in differences by +0.89 SR points (95 % CI [-1.60, +3.89], p = 0.53); per condition
  +0.04 (`tok_block_rgb_04`, p = 0.98) and +1.73 (`tok_noise_rgb_04`, p = 0.16).  Both arms drop
  ~13.5 points on `tok_block_rgb_04` against the shipped arm's 9.0 from a higher clean baseline.
- **P2 tracking predictions recorded before the data (`docs/results.md` 6.22).**  With the
  corrected geometry in hand (identity-residual arms *reduce* the damaged-token error, `post_norm`
  arms raise it 32x), the tracking contrasts are pre-registered: A-vs-D composite positive on the
  token conditions, C/D above A/B on clean, A-vs-B and C-vs-D close to each other (the gate is
  inert), and -- if the primary prediction fails despite that geometry -- the conclusion is that
  token-level repair does not propagate through the fusion and head rather than that the
  parameterisation is worthless.
- **P2 tracking, first arm (`docs/results.md` 6.22, partial).**  Arm A -- the shipped `post_norm`
  recipe retrained with the current code -- tracks **5.18 SR points worse than the shipped
  checkpoint on clean** (paired, p = 0.0024) and 9.64 points worse under `tok_block_rgb_04`
  (p = 0.0015), matching its 34.4x-overshooting update geometry: the output LayerNorm leaves the
  update scale unconstrained, and two runs of the same configuration land 20x apart in scale and
  5 SR points apart in tracking.  Also recorded: `tok_noise_rgb_04` leaves A *above* its clean SR
  (0.306 vs 0.290), and A's own clean run needed the box clamp on 43 frames while neither
  corruption condition did.
- **Correction: the branch repairs its own configuration.**  `docs/results.md` 6.21.  The
  "near no-op / inflates the error" reading of 6.7.1 and 6.20 came from probes run with
  `--corrupt-target rgb`, while training erases **both** modalities (`corruption.target: both`).
  With the training setting the same checkpoint reports `recovery_gain` **+0.0553** (free crops)
  and **+0.0499** (fixed), positive on 83 % of sequences.  Four other hypotheses (closed-loop
  drift, held-out domain, fp16/AMP, crop centring) were each tested and rejected before the
  configuration was checked, using `tools/trajectory_replay.py --recovery` and the new
  `tools/fp16_recovery_check.py`.  What stands is specificity: the branch repairs the configuration
  it was trained on, not rgb-only erasure, not a non-zeroing family, not image-level degradation.
- **Retracted claim, kept visible:** I first wrote in 6.21 that the image-level corruptions
  declared in the config are never applied in training (inferred from a grep of `Trainer.train()`
  that found no `apply_image_corruption`).  Measurement refutes it: the *dataset* applies them to
  the search frame before cropping, while the template stays pristine (`search_rgb` differs by
  mean |delta| 1.306, `template_rgb` is bit-identical).  The correction reverses the reading of
  6.5's image conditions -- they apply one corruption type while training applies four at once, so
  they are milder than training rather than out of distribution -- and the property is now pinned
  by `tests/unit/test_dataset_corruption.py`.
- `tools/fp16_recovery_check.py` -- runs the *training* loss on real batches in fp32, fp16 and
  eval mode and prints the recovery terms of all three, which is how the fp16/mode hypotheses for
  the train/eval recovery gap were ruled out.
- `tools/trajectory_replay.py --recovery [--fixed-gt]` -- collects the per-frame recovery
  diagnostics of the clean, free-crop, fixed-crop and (optionally) ground-truth-crop arms and
  reports `recovery_gain` per arm, with the uncorrupted arm correctly reported as undefined rather
  than as 0.
- **Step-1 (geometry) readout, all four arms, corrected (`docs/results.md` 6.10).**  With each
  arm's runtime overrides applied, the table reverses for C and D: `gain vs zero` is **positive**
  for both (+0.0145 C, +0.0151 D) at `applied / oracle` 1.015 and 1.004 -- the identity-residual
  arms *reduce* the damaged-token error (D: 0.1856 -> 0.1705, -8.1 %, within 0.0006 of the best
  global scale of their own direction) -- while the `post_norm` arms overshoot by 34-35x and push
  it up by 5.8-5.9 L1.  C's learned gate opens to a median 0.840 (shipped 0.004) with
  `rho(gate, a*)` positive, and the gate remains inert in every arm.  The previous version of this
  table had C and D probed through a `post_norm` decoder (overrides omitted) and concluded the
  opposite ("do no harm, useless direction"); the invalid artifacts are kept as
  `outputs/step_size_probe_p2_{B,C,D}_invalid_arch.json`.
- **Step-1 (geometry) readout of the P2 grid, partial, `docs/results.md` 6.10.**  The bounded-step
  arm applies **1.065x** the ideal step against **1.677x** for the shipped checkpoint and
  **34.4x / 28.7x** for the retrained `post_norm` arms, whose update hurts the token error by
  4.8-5.8 L1 points (shipped 0.107, D 0.012).  Retraining `post_norm` twice with the same
  configuration lands 20x apart in scale, because the output LayerNorm absorbs it -- the strongest
  case for the identity-residual output, visible without any tracking run.  Arm D's *direction*,
  however, carries no gain (`a* = 0` for 43 % of damaged tokens, oracle L1 = zero-step L1), so it
  achieves "do no harm" rather than "repair".
- **Arm C aborted under AMP; arm D (same parameterisation, gate = 1) did not.**  A CUDA device-side
  assert in `BCELoss` (`input_val >= 0 && <= 1`) after ~2 200 steps, i.e. a NaN probability from an
  fp16 overflow in the un-normalised residual path.  `NeuralBPDecoder._step` now replaces a
  non-finite update with zero and counts it (`decoder_nonfinite_steps`, warned in the training log,
  unit-tested), which makes the event diagnosable instead of fatal, and arm C is retrained in fp32
  with the guard in place.  `tools/compare_step_probes.py` no longer throws away the other arms
  when one artifact is missing.
- Provenance recorded in `docs/results.md` 6.10: the four P2 arms are mutually comparable (one
  session, one code revision, one seed), but **arm A is not a bit-identical replay of the shipped
  `ab_full`** (final loss 3.1390 vs 3.2492, `track` 0.4826 vs 0.4205) because the corruption and
  sampling RNG stream shifted between the two runs.  No result is therefore reported as A vs
  `ab_full`; the shipped checkpoint is only the historical reference for the step-probe ratios.
- The P2 grid's analysis is pre-registered in `docs/results.md` 6.10 before its runs finish:
  update geometry first (step-probe ratios), then two single-factor contrasts at m = 1 (A vs B for
  the gate, C vs D for the same parameterisation with and without it), then the primary A-vs-D
  composite split by condition -- because a gain that appears only on the erasing family would not
  support the bounded-step story of 6.7.1 -- with `clean` as a level check and the divergence rate
  as a validity flag.
- The four P2 arms were verified before training (decoder parameter counts, the exact
  zero-initialisation of the residual projection, and the resolved config of each `--override`
  list): A/B 4.9277 M, C/D 5.5199 M, i.e. the identity-residual arms carry 12 % more decoder
  capacity, which is recorded in `docs/results.md` 6.10 as a confound rather than glossed over.
- `model.decoder_output` + `model.residual_clip` + `model.gate_always_one` — the decoder can
  now be built with the LayerNorm *inside* the update branch and an un-normalised output
  (`identity_residual`), a zero-initialised residual projection (the model starts at the
  identity), a fixed L2 budget on each step, and without the learned severity gate.  The
  default `post_norm` / `0.0` / `false` reproduce the shipped behaviour bit for bit, and the
  default path is locked by a unit test.
- `tools/calibrate_residual_budget.py` now also reports the **L1 deviation quantiles** of damaged
  and healthy tokens and the suggested `detect_deviation_cm` for the P4 target, so that squash
  scale is measured rather than assumed.
- `tools/calibrate_residual_budget.py` — takes a fixed high quantile of the *ideal* residual
  `x* - x` on a pre-declared calibration split and prints the `model.residual_clip` to use
  (19.045 for RGB token block erasure at ratio 0.2 on the 60-sequence development split), so
  the bound is an engineering budget rather than a number tuned on the evaluation set.
- `tools/horizon_probe.py` — per-sequence IoU trace, first frame after which the rolling mean
  IoU never recovers above 0.5, and the loss-time distribution, because an SR average cannot
  distinguish slow drift from a tracker that was never on the target.
- `codetrack/cli/eval.py --sequence-list` and `Trainer.evaluate(sequence_list=...)` —
  evaluate an explicit sequence list; the default `None` keeps the historical
  "first N of the subset" path bit-identical.
- `model.mlp_hidden` (convolution-free config key, default `0`) — names the MLP baseline's
  hidden width instead of always solving it from the BP parameter count. `0` keeps the
  auto/parameter-matched width, so the baseline path is unchanged; an already-trained MLP arm
  (trained at 2048 hidden with the earlier formula, 6.307 M decoder parameters) can now be
  rebuilt and loaded with `model.mlp_hidden=2048`.
- `tools/validate_sequences.sh`: `token`-level conditions `tok_block_tir_04` and
  `tok_block_both_04`, completing the ratio 0.2 / 0.4 curve for RGB-only, TIR-only and
  dual-modality corruption.

### Fixed
- `codetrack/utils/checkpoint.py`: `load_checkpoint` now refuses to load a checkpoint whose
  stored `model.decoder_mode` differs from the model being built. An `off` arm loaded into a
  `bp` model left 16 message-passing tensors at their random initialisation (and the MLP arm
  14 + 8 unexpected) while `strict=False` reported it only in a list nobody read, so an
  evaluation could silently measure a randomly wired decoder. `Trainer.evaluate` also logs a
  warning whenever a load reports missing or unexpected keys.
- `tools/validate_sequences.sh`: takes `OVERRIDES` for the model config a checkpoint requires
  and records it in `run_manifest.json`.
- `tools/summarize_paired_conditions.py`: the difference-in-differences sign now matches the
  reviewer's convention (`D = drop_B - drop_A`, positive = arm A degrades less; it used to be
  reported the other way round), and the Holm-Bonferroni family now spans the condition
  comparisons **and** the `--interaction` contrasts in one correction instead of correcting
  each separately.
- `tools/p2_grid_report.sh` resolves its interpreter properly (`PYTHON_BIN`, else `python3`), which
  is how its first invocation failed with `python: command not found`.
- `tools/p2_grid_report.sh` — runs the pre-registered four-arm recipe in one command (gradient
  conflict, update geometry against the shipped decoder, divergence rates, the two single-factor
  contrasts at m = 1, the primary composite split by condition, the clean level check and the
  IoU-endpoint check), writing every table to `outputs/p2_report/`.
- `tools/gradient_conflict_report.py` — parses the `[grad] cos(L_track, L_correct)` series each
  training run already logs, so a worse-tracking arm can be told apart from an arm whose two
  losses fight.  Measured: shipped +0.063 (min -0.250), `p2_A` +0.093, `p4_deviation` +0.128, all
  mildly aligned.
- `tools/family_report.py` — one table per arm and corruption family (reliability AUROC per
  modality, syndrome AUROC, locator precision, recovery gain and the raw error magnitudes), with
  the primary endpoint of `docs/results.md` 6.18 compared against the mask-trained baseline, so
  the P4 decision rule is executed by a command rather than by hand.
- `tools/clamp_audit_report.py` — pre-fix vs clamped SR/PR/NPR/IoU side by side for the
  re-run conditions, with the clamp counts, so the effect of the crop bound on already published
  numbers is measured rather than assumed; a condition that has not been re-run prints `missing`.
- **Queue deadlock (process bug, fixed).**  The waiting scripts used `pgrep -f` on strings such as
  `validation_queue3`, and those strings also occurred in the command line of the `bash -c`
  wrappers that had created the scripts -- so every waiter matched a stale wrapper of itself, all
  four queues blocked, and the GPU sat idle while they waited for each other.  The wrappers are
  gone and the remaining work now runs from one serial queue (`/tmp/final_queue.sh`) that waits on
  nothing: stages run in order, and the script itself is created with a file-writing tool so its
  text never becomes a command line.  A nested-heredoc slip in the same command leaked script text
  into the shell (only `echo`/`python` fragments ran, nothing was created or launched); the serial
  script is now written directly to avoid that class of mistake too.
- `tok_block_both_04` (both modalities erased at 40 %) finished after the fix and is a
  **closed-loop breakdown, not a degradation**: 8 928 of 10 244 frames (87.15 %) needed the box
  clamp, the worst sequences on 190-197 of 200 frames, and SR 3.70 % against a clean 33.95 %.
  Recorded in `docs/results.md` 6.5 and 6.15 as a failure-mode measurement and kept out of the
  calibrated condition set.
- **Unbounded crop window (found while re-running `tok_block_both_04`).**  `_crop_square`
  cropped the *predicted* side with no bound, so a diverged box made `np.pad` allocate a window
  the size of the prediction -- a 40000-pixel side costs 6.7 GB, and one 60-sequence evaluation
  grew past 21 GB with the GPU idle and no result.  `_crop_square` now clamps the side to 8x the
  frame's longer edge and the centre onto the frame, and `Trainer.infer_sequence` clamps every
  predicted box (`clamp_box`, at most 4x the frame, finite, positive) while *counting and
  logging* what it touched (`box_clamps` per sequence, `n_box_clamped_frames` in the summary).
  Runs completed before the fix did not record clamps; a separate audit re-runs the strongest
  token conditions into `outputs/validation_v1/ab_full_clamped/` for comparison.
- `tools/horizon_probe.py`: the loss-time event is now forward-decidable -- the first frame of
  `patience` consecutive frames below the threshold, with un-failed sequences right-censored and
  a truncated mean survival time, plus the recovery rate.  The old "never recovers again" number
  is kept as a descriptive statistic (`loss_frame`) but is no longer the event definition.
- `tools/gate_information.py`: the scattered severity gate was reported as "what the decoder
  multiplies", which it is not -- the applied coefficient is `(1 - r) * gate`, and the
  selector's preference for damaged tokens concentrates the neutral 1s on healthy tokens, so
  the old AUC could not be read as "the repair is suppressed". `applied_coefficient` is now
  scored explicitly.
- `docs/results.md`: the damaged-token gain is quoted with **both** denominators (2.49 %
  overall vs 15.6 % relative to the norm-only baseline); the strength screen reports the
  per-sequence and pooled relative drops (21.64 % vs 26.60 % for RGB 0.4) and notes that the
  pre-registered arm is the frozen no-decoder one; the TIR Holm-adjusted p-value (0.079) is
  stated next to the raw 0.040; the "gate suppresses the repair" and "label memorisation"
  readings are retracted with the corrected wording.
- `docs/architecture.md` section 13.5: "learned sparse H" -> "learned edge weights on a fixed,
  geometrically constructed sparse support" (the support is a buffer the optimiser cannot
  move), and section 11.7 gained a caveat that the joint template/search self-attention puts
  clean-search information into the parity reference -- an unmeasured shortcut.
- `tools/recovery_probe.py`: RGB/TIR pairing no longer assumes `infrared/i<rest>`. The
  resolver decides **one strategy per sequence** (identical name, modality-prefix swap,
  modality letter as a suffix, identical frame number, or positional alignment that is
  only accepted when both listings are provably index-aligned) and refuses a sequence
  instead of mixing rules -- which had paired an offset sequence half by name and half not
  at all. On the standard 20-sequence selection the old rule silently dropped 72/240
  frames; over `testingset` it left 16 257/220 703 frames unpaired, and it skipped any
  sequence whose visible frames are not `v<digits>.jpg`. Probes now report the pairing
  counts, the usable frame count and every skipped item.
- `codetrack/engine/trainer.py`: image-level corruption conditions crashed with
  `KeyError: 'corrupted_rgb'` in `_collect_diagnostics`, and `evaluate` swallowed it as
  "skip <sequence>" -- every RGB low-light, RGB occlusion and TIR crossover run scored
  zero sequences. The clean-reference pass also reused the *degraded* frame, which made
  the syndrome AUROC of those conditions 0.5 by construction; it now uses the pristine
  frame.
- `tools/diagnostics.py`: `diagnostics.json` stores the invocation it was produced with.
  The same checkpoint scores 66.60 SR at `--frames 10` and 50.90 at `--frames 30`, and the
  artifact previously could not say which setting it came from.

## [0.1.0] - 2026-10-03

### Added
- Repository initialized.
