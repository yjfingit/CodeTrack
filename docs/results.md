# Results

> Placeholder. Every entry must carry the config path, commit hash, seed and log path.
> Rule: no number without a reproducible command (see `docs/reproducibility.md`).

## 6. Review-round diagnostics (base commit `bdd3c7b`, 2026-10-03)

Every number here comes from a committed tool plus an artifact under `outputs/`. Unless
stated otherwise the checkpoint is `outputs/ab_full/final.pth`: 10 epochs, `decoder_mode=bp`,
`h_links_per_check=24`, `syndrome_use_obs_energy=false`, trained under
`configs/experiment/lasher_vitb_corrupt.yaml` with `tok_burst_erase` at ratio 0.2.
`tools/recovery_probe.py --checkpoint outputs/x_full/final.pth` results are also listed
because the earlier summary used that checkpoint; it was trained with
`h_links_per_check=32` and `tok_random_erase`, and the incidence actually used at
inference is the one stored in the checkpoint (634 vs 625 edges, verified by loading).

### 6.1 RGB/TIR frame pairing (`tools/pairing_audit.py`)

The probe used to pair `infrared/` frames with `"i" + name[1:]`, which silently dropped
whole sequences; the resolver now decides one strategy per sequence and never mixes rules.

| subset | sequences | frames | paired (now) | paired (old rule) |
|---|---:|---:|---:|---:|
| `testingset` | 245 | 220 703 | **220 703 (100.000 %)** | 204 446 (92.6 %) |
| `trainingset` | 979 | 514 081 | **512 817 (99.754 %)** | 453 708 (88.3 %) |

* Strategy chosen: 245/245 `name` in `testingset`; 977 `name`, 1 `position`
  (`orange`, camera-timestamp IR names), 1 refused (`bowblkboy1-quezhen`: IR numbering has
  17 gaps, so positional alignment is not verifiable and the sequence is reported instead
  of half-paired).
* `name` vs sorted-order (the convention the tracker itself uses) disagreements: **0**.
* On the 20-sequence / 12-frame probe selection the old rule dropped 72 of 240 search
  frames (6 sequences, all `000001.jpg`-style names); the new one pairs 240/240.
* Command: `python tools/pairing_audit.py --out outputs/pairing_audit.json`.

### 6.2 Four-cell recovery probe on real frames (`tools/recovery_probe.py`)

20 sequences x 12 frames = 240 frames, RGB-only corruption, ratio 0.2, severity 0.4,
seed 0, `testingset`; 0 skipped. `v` = corrupted input, `ln_v` = output LayerNorm without
messages, `v_plus_d` = gated update before the output LayerNorm, `ln_v_plus_d` = full
output. Errors are mean L1 against the clean token.

`outputs/ab_full/final.pth` (`outputs/recprobe_ab_full_pairfix/probe.json`):

| regime | cell | damaged RGB | healthy RGB | RGB target | whole TIR |
|---|---|---:|---:|---:|---:|
| block | `v` | 0.1968 | 0.0000 | 0.1418 | 0.0000 |
| block | `ln_v` | 0.2274 | 0.2171 | 0.2619 | 0.2055 |
| block | `v_plus_d` | 0.3253 | 0.0284 | 0.2073 | 0.0952 |
| block | `ln_v_plus_d` | **0.1919** | 0.1610 | 0.2451 | 0.1166 |
| block_replace | `v` | 0.2293 | 0.0000 | 0.1692 | 0.0000 |
| block_replace | `v_plus_d` | 0.2394 | 0.0232 | 0.1857 | 0.0942 |
| block_replace | `ln_v_plus_d` | 0.3496 | 0.1705 | 0.2986 | 0.1182 |
| random | `v` | 0.1778 | 0.0000 | 0.0615 | 0.0000 |
| random | `v_plus_d` | 0.3182 | 0.0268 | 0.1049 | 0.0973 |
| random | `ln_v_plus_d` | **0.1718** | 0.1642 | 0.2137 | 0.1174 |
| feat_noise | `v` | 0.1081 | 0.0000 | 0.0208 | 0.0000 |
| feat_noise | `v_plus_d` | 0.1316 | 0.0227 | 0.0352 | 0.0934 |
| feat_noise | `ln_v_plus_d` | 0.2425 | 0.1720 | 0.1996 | 0.1181 |

`outputs/x_full/final.pth` (`outputs/recprobe_x_full_pairfix/probe.json`): the same four
cells are worse than `v` on damaged tokens in every regime -- `v_plus_d` 0.7074 / 0.2916 /
0.7051 / 0.1983 and `ln_v_plus_d` 0.2117 / 0.2966 / 0.1896 / 0.2064 against
`v` = 0.1968 / 0.2293 / 0.1778 / 0.1081 for block / block_replace / random / feat_noise
(post-norm gains -7.6 %, -29.4 %, -6.6 %, -90.9 %).

Reading, and what it does not say:

* For `ab_full` the post-norm result on **block** (0.1919 vs 0.1968) and **random**
  (0.1718 vs 0.1778) is a small improvement; the energy-matched replacement and additive
  noise regimes are clearly worse. So "the decoder never helps" is not supported by the
  current checkpoint, and "the decoder repairs every corruption type" is not either.
* `v_plus_d` is not comparable to `v` and `ln_v_plus_d` is not comparable to `v`: the output
  LayerNorm moves even an *untouched* token by 0.22-0.33 L1, and `masked_recovery` -- the
  objective the model is trained on -- also scores the post-norm output against the clean
  token. Only `ln_v_plus_d` vs `ln_v` is a same-space comparison.
* The four cells are a feature-space diagnosis. They are **not** a tracking result: deleting
  the output LayerNorm changes the head's input distribution, so PR/SR must be re-trained
  before any comparison. See section 6.4 for the direction/gate decomposition.
* Commands:
  `python tools/recovery_probe.py --checkpoint outputs/ab_full/final.pth --out outputs/recprobe_ab_full_pairfix/probe.json --sequences 20 --frames 12 --ratio 0.2 --seed 0`
  (and the same with `outputs/x_full/final.pth`).

### 6.3 Incidence shuffle: tracking unchanged, localization destroyed

`tools/diagnostics.py`, 5 sequences x 10 frames, `--target rgb`, `tok_block_erase`, ratio
0.2, severity 0.4 (`outputs/diagnostics_shuffle_confirm_f10/diagnostics.json`). This is the
exact invocation that produced the earlier 66.60 -> 66.70 pair, and it reproduces
bit-identically:

| probe | PR | SR (AUC) | NPR | syndrome AUROC | locator P@5 (chance 0.199) | density rho |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 78.00 | 66.60 | 76.00 | 0.401 | 0.796 | 0.807 |
| shuffle-incidence | 78.00 | 66.70 | 76.00 | 0.330 | 0.332 | 0.544 |

At 25 sequences x 30 frames (`outputs/diagnostics_shuffle_25seq/diagnostics.json`) the same
pattern holds: SR 50.45 -> 50.72 and PR 72.13 -> 72.40 while locator P@5 falls 0.760 ->
0.266 (chance 0.199) and syndrome AUROC 0.395 -> 0.324.

Interpretation: the tracking metrics of this checkpoint do not depend on the incidence,
while the localization/density readouts do. A "no tracking change" shuffle result is
therefore not evidence that `H` is unused -- it is evidence that the *tracking* path is not
where `H` currently acts. The intervention is applied after loading the checkpoint
(`run_probe`), so it is measured on the trained model.

> `--frames` matters: the recorded 66.60/66.70 pair is a 10-frame evaluation. The same
> checkpoint evaluated over 30 frames scores 50.90 SR / 68.00 PR, i.e. the horizon changes
> the number, which is exactly why the invocation is now stored inside `diagnostics.json`.

### 6.4 Where the update goes wrong: gate policies and update direction

`tools/gate_probe.py`, 20 sequences x 12 frames = 240 frames, the same corruption sample for
every policy. `zero` passes a full `B x N` zero gate and is a falsifiable identity check;
`full` opens the gate everywhere while `(1 - r)` still applies; `oracle` opens it exactly on
the corrupted variables. `cos` is the mean per-token cosine between the applied update
`pre_norm - v` and the ideal update `clean - v` on damaged tokens; `|d|/|d*|` is the
applied/ideal update-norm ratio. `ln_v` is the same for every policy (it is the output
LayerNorm applied to the uncorrected input), so a policy can only be read against `ln_v`.

`outputs/ab_full/final.pth` -- the current recipe, `outputs/gate_probe_ab_full.json`:

| regime | policy | damaged `v` | `v_plus_d` | `ln_v` | `ln_v_plus_d` | cos | \|d\|/\|d*\| |
|---|---|---:|---:|---:|---:|---:|---:|
| block | learned | 0.1968 | 0.3253 | 0.2274 | **0.1919** | +0.631 | 2.22 |
| block | zero | 0.1968 | **0.1968** | 0.2274 | 0.2274 | 0.000 | 0.00 |
| block | full | 0.1968 | 0.7886 | 0.2274 | **0.1881** | +0.608 | 10.09 |
| block | oracle | 0.1968 | 0.7864 | 0.2274 | 0.1880 | +0.608 | 10.06 |
| block_replace | learned | 0.2293 | 0.2394 | 0.3932 | 0.3496 | -0.040 | 0.36 |
| block_replace | zero | 0.2293 | **0.2293** | 0.3932 | 0.3932 | 0.000 | 0.00 |
| block_replace | full | 0.2293 | 0.2458 | 0.3932 | 0.3221 | -0.040 | 0.58 |
| block_replace | oracle | 0.2293 | 0.2458 | 0.3932 | 0.3221 | -0.040 | 0.58 |

`outputs/x_full/final.pth` -- the older recipe (`h_links_per_check=32`,
`tok_random_erase`), `outputs/gate_probe_x_full.json`:

| regime | policy | damaged `v` | `v_plus_d` | `ln_v` | `ln_v_plus_d` | cos | \|d\|/\|d*\| |
|---|---|---:|---:|---:|---:|---:|---:|
| block | learned | 0.1968 | 0.7074 | 0.2011 | 0.2117 | +0.588 | 7.17 |
| block | zero | 0.1968 | **0.1968** | 0.2011 | 0.2011 | 0.000 | 0.00 |
| block | full | 0.1968 | 1.1427 | 0.2011 | 0.2215 | +0.587 | 11.51 |
| block | oracle | 0.1968 | 1.0386 | 0.2011 | 0.2266 | +0.586 | 10.39 |
| block_replace | learned | 0.2293 | 0.2916 | 0.4978 | 0.2966 | -0.026 | 1.58 |
| block_replace | full | 0.2293 | 0.3145 | 0.4978 | 0.2780 | -0.026 | 1.95 |

Reading:

* `zero` reproduces the corrupted input **exactly** (`identity_check: true` in both files), so
  the update path and its gate plumbing are wired correctly; `ln_v_plus_d == ln_v` there by
  construction. This is the "gate = 0 leaves the token untouched" invariant the review asked
  for, and it holds.
* **The two checkpoints disagree**, which is why the earlier single-checkpoint conclusion does
  not generalise. `ab_full` aligns the update with the ideal one (cos +0.63) at 2.2x its
  magnitude and slightly improves the damaged-token error in the model's own post-norm space
  (0.1919 vs 0.1968 input, 0.2274 norm-only). `x_full` overshoots by 7-12x and is worse than
  the input in every regime recorded here.
* Opening the gate is not the fix. For `ab_full`, `full`/`oracle` multiply the update by ~4.5x
  (ratio 2.2 -> 10.1) and move the pre-norm error from 0.33 to 0.79 while the post-norm error
  barely changes (0.1919 -> 0.1881). The binding constraint is the update *scale*, not the
  gate *level*.
* The learned severity gate does not separate damage in either checkpoint: `ab_full`
  `damaged_mean = 0.2253` vs `healthy_mean = 0.2420`, `x_full` 0.6812 vs 0.6764 (selected
  nodes are half the 256 variables, `selected_fraction = 0.5000`). Whatever scales the update
  down for reliable tokens, it is `(1 - r)`, not this gate.
* Energy-matched block *replacement* is a different failure: the direction is orthogonal
  (cos -0.04) at 0.36x the ideal magnitude, and no gate policy repairs it. Direction, not
  scale.
* Consequence for the next structural experiment (deliberately **not** applied yet): keep the
  identity path and bound the update -- pre-norm inside the update branch, a zero-initialised
  final residual projection, and a residual target rather than a post-norm reconstruction.
  Because `L_correct`/`masked_recovery` score the post-norm output against clean tokens, an
  8-12x pre-norm overshoot costs the loss little; removing the output LayerNorm *without* that
  change would expose the overshoot instead of improving the tracker.
* Commands:
  `python tools/gate_probe.py --checkpoint outputs/ab_full/final.pth --sequences 20 --frames 12 --ratio 0.2 --regimes block,block_replace --gates learned,zero,full,oracle --out outputs/gate_probe_ab_full.json`
  and the same for `outputs/x_full/final.pth`.

### 6.5 Stratified long-horizon validation (partial)

**Status: partial.** 60 `testingset` sequences x up to 200 frames, 12 per attribute stratum
(`outputs/validation_split_v1/`), three checkpoints, nine conditions. This entry records what
had completed when it was written; the remaining conditions and the ratio-0 control were still
running, so nothing here is final.

Clean baseline (`outputs/validation_v1/ab_full/clean/metrics.json`):

| stratum | n | clean SR (AUC) | median |
|---|---:|---:|---:|
| unoccluded | 12 | 0.478 | 0.496 |
| partial occlusion | 12 | 0.343 | 0.359 |
| total occlusion | 12 | 0.330 | 0.189 |
| thermal crossover | 12 | 0.300 | 0.240 |
| low illumination | 12 | 0.246 | 0.112 |
| **all** | **60** | **0.340** | 0.291 |

The same checkpoint scores 0.72 SR on the 20-sequence / 12-frame convenience sample the
earlier curves used, so the two are not comparable: the stratified split is harder *and* 200
frames is a much longer horizon. `tools/horizon_probe.py` reports the mechanism (loss-time
distribution and per-frame IoU trace). A three-sequence pilot showed the typical failure is
not slow drift -- on `basketballathand` IoU falls 0.78 -> 0.23 within seven frames while the
target moves ~5 px/frame, while `mandownstair` (slow motion) tracks at 0.80 for all 200
frames. The full 60-sequence loss-time run is queued.

Paired clean-vs-corrupt deltas, 60 sequences, positive = corruption hurt
(`outputs/validation_v1/partial_ab_full.json`):

| condition | clean SR | corrupt SR | drop (pts) | 95% CI (pts) | p(t) |
|---|---:|---:|---:|---|---:|
| `tok_block_rgb_02` | 0.3395 | 0.3051 | **+3.44** | [0.98, 6.33] | 0.015 |
| `tok_block_rgb_04` | 0.3395 | 0.2492 | **+9.03** | [4.92, 13.54] | 0.0001 |
| `tok_block_tir_02` | 0.3395 | 0.3868 | **-4.74** | [-9.13, -0.37] | 0.040 |
| `tok_block_both_02` | 0.3395 | 0.3531 | -1.36 | [-4.43, 1.65] | 0.389 |
| `tok_block_rgb_02`, reference-tracking subset (n=32) | 0.5611 | 0.5003 | +6.08 | [1.57, 11.11] | 0.019 |
| `tok_block_rgb_04`, reference-tracking subset (n=32) | 0.5611 | 0.4065 | +15.45 | [8.44, 23.11] | 0.0003 |
| `tok_block_tir_02`, reference-tracking subset (n=32) | 0.5611 | 0.5799 | -1.88 | [-8.11, 4.63] | 0.569 |

Pre-registered strength screen (`--screen-run ab_full`; the pre-registration is defined on the
frozen *no-decoder* arm, so this is the same computation on the decoder arm as a calibration
check): relative drop 20-40 % with 60-80 % of the clean SR retained, and an interval that
excludes zero.

| condition | rel % | retained % | interval excludes 0 | verdict |
|---|---:|---:|---|---|
| `tok_block_rgb_02` | 7.88 | 89.87 | yes | too weak |
| `tok_block_rgb_04` | 21.64 | 73.40 | yes | **pass** |
| `tok_block_tir_02` | -54.32 | 113.95 | yes | helps (not a degradation) |
| `tok_block_both_02` | -25.80 | 104.01 | no | n.s. |

So on this split, **ratio 0.4 is the calibrated reference level for RGB token block erasure and
0.2 is too weak**; the TIR and dual-modality levels are not degradations at all under this
checkpoint. That is a calibration result, not a method result, and it is reported as the
whole curve rather than as the level that happens to favour one arm.

Per-stratum paired drop (points, positive = corruption hurt):

| condition | low illum | partial occl | thermal cross | total occl | unoccluded |
|---|---:|---:|---:|---:|---:|
| `tok_block_rgb_02` | +3.31 | +4.19 | +0.49 | +2.22 | +6.98 |
| `tok_block_rgb_04` | +10.24 | +7.62 | +8.38 | +2.92 | +16.00 |
| `tok_block_tir_02` | +1.69 | -1.39 | -3.77 | -9.10 | -11.10 |
| `tok_block_both_02` | -4.42 | -1.34 | +4.21 | -5.57 | +0.32 |

* 20 % RGB block erasure costs 3.4 SR points overall and 6.1 points on the sequences the clean
  arm actually tracks: a real but small degradation on a 34-point baseline, positive in every
  stratum.
* 20 % **TIR** block erasure *improves* the full-set SR by 4.7 points, most of it on sequences
  whose clean SR is near zero; on the tracked subset the effect is +1.9 points with an
  interval containing zero. This is not read as "TIR corruption is good": the leading
  hypothesis is that the TIR branch contributes harmfully to the fusion on this checkpoint,
  which the remaining cells (TIR ratio 0.4, TIR crossover, TIR-only arm) can test.
* A **ratio-0.0 control** (same evaluation path, one erased token per frame, which cannot move
  tracking) is queued to rule out the diagnostics pass -- corrupted conditions run an extra
  clean-reference forward -- as the cause of the TIR sign.
* These are not population estimates: 60 stratified sequences, paired intervals over
  sequences, one checkpoint, one severity, one 200-frame horizon.
* Commands:
  `python tools/select_validation_sequences.py --out outputs/validation_split_v1 --n 60 --seed 0`;
  `bash tools/validate_sequences.sh` (see `docs/corruption_protocol.md` section 6);
  `python tools/summarize_paired_conditions.py --run ab_full=outputs/validation_v1/ab_full --strata-manifest outputs/validation_split_v1/manifest.json`.

### 6.6 Ablation-arm provenance, and a load guard that came out of it

The first no-decoder matrix of this round was **discarded before any number was reported**,
because the arm was evaluated with the wrong model: `outputs/ab_nodec_corr/final.pth` was
trained with `model.decoder_mode=off`, and loading it into the config's default `bp` model
left 16 message-passing tensors at their random initialisation. `strict=False` returns those
keys in a report nobody read, so the run looked healthy and measured a randomly wired
decoder. The same check on `outputs/ab_mlp_corr/final.pth` shows 14 missing and 8 unexpected
decoder tensors.

Two guards now make that failure mode impossible to reach silently:

* `codetrack/utils/checkpoint.py:load_checkpoint` **raises** when the checkpoint's stored
  `model.decoder_mode` differs from the model being built, and says which override to pass;
* `Trainer.evaluate` logs a warning whenever a load reports missing or unexpected keys;
* `tools/validate_sequences.sh` takes `OVERRIDES="model.decoder_mode=..."`, so the override
  is part of the recorded `run_manifest.json` instead of living in a shell history.

A second provenance fact has to be stated with the numbers: the three ablation arms were
trained at 20:10-20:23, before the working tree's last edits to
`codetrack/models/decoder/bp.py` (21:48) and `codetrack/models/codetrack.py` (21:40). The
clearest symptom is the MLP arm's width: it was trained with the pre-fix auto formula
(2048 hidden, 6.307 M decoder parameters -- the over-parameterised control the review
flagged), while the current formula derives the parameter-matched width (1600 hidden: 4.928 M
for `bp`, 4.929 M for `mlp`). The width is now a config key, `model.mlp_hidden` (default `0`
= the auto/parameter-matched value, so the baseline path is unchanged), and with
`model.mlp_hidden=2048` the stored arm loads with **0 missing / 0 unexpected** keys. It stays
labelled **over-parameterised**; a matched MLP arm has to be retrained together with the
other arms before the "graph vs MLP" question is answered. The numbers in section 6.5
therefore describe *these trained weights under the current inference code*, and the retrain
is on the critical path for any claim about the current architecture.

### 6.7 What the gate inputs know, and what the gate does with it

`tools/gate_information.py`, 4 sequences x 12 frames = 48 frames under RGB token block
erasure at ratio 0.2 (`outputs/gate_information_ab_full.json`; a 16-frame pilot shows the same
pattern, so it is not a small-sample accident). ROC AUC of each internal score against the
injected mask:

| score | AUROC vs injected mask | what it means |
|---|---:|---|
| `1 - reliability_rgb` | **1.000** | the reliability head is a perfect mask classifier -- it is trained with `BCE(r, 1 - mask)` |
| `locator_scattered` (`H^T s`) | 0.853 | the locator ranks damaged tokens well; also mask-supervised |
| `selection_of_damage` | 0.812 | the selector preferentially puts damaged tokens into the graph |
| `gate_within_selected` | **0.392** | the learned severity gate is *lower* on damaged tokens than on healthy ones |
| `effective_gate_rgb` (unselected keep 1) | **0.147** | what the decoder actually multiplies: the correction is suppressed exactly where it is needed |
| `syndrome_density_rho` (within frame) | 0.764 | the syndrome tracks each check's realized corruption density |

Read with section 6.4 this is one mechanism, not two findings:

1. The model knows where the damage is (`1 - r` AUC 1.0), the selector focuses the graph on it
   (0.812), and `(1 - r)` already gates correctly by itself (1 on damaged tokens, 0 on healthy).
2. On top of that the learned severity gate is *anti*-informative (0.392 within the selected
   nodes), so the net multiplier `(1 - r) * gate` is ~4-5x smaller on damaged tokens than the
   reliability factor alone would give.
3. That is the rational thing to learn in the current output space: forcing the gate open
   (`full`/`oracle`, section 6.4) drives the *pre-norm* error from 0.33 to 0.79 while the
   post-norm error barely moves, because `L_correct` / `masked_recovery` compare the post-norm
   output against pre-norm clean features, and the output LayerNorm moves even an untouched
   token by 0.22-0.33 L1. Suppressing the update is the cheapest way to lower that loss.

So the failure is **not** "the model cannot find the damage", and **not** "the gate is too
small": the correction branch is optimised in a space where the output LayerNorm absorbs the
scale error, and the gate rationalises that by shrinking the repair. Two consequences:

* Detection/localization AUROC is not evidence of error-correction capability -- those heads
  are supervised with the synthetic mask, so `1 - r` at AUC 1.000 is label memorisation.
  Report them as "predicts the injected mask" and pair them with a mask-independent check.
* Any claim that the Tanner structure improves tracking has to come from the paired
  difference-in-differences on held-out sequences (`--interaction`), not from these AUCs.

## 1. Clean evaluation (no injected corruption)
| Tracker | Backbone | Params (M) | GFLOPs | FPS | Dataset | PR | SR (AUC) | NPR |
|---|---|---:|---:|---:|---|---:|---:|---:|
| ViPT (official) | ViT-B/16 | — | — | — | LasHeR | — | 52.46 | — |
| CodeTrack (ours) | ViT-B/16 | — | — | — | LasHeR | — | — | — |

## 2. Corruption robustness sweep

Severity in `{mild, moderate, severe, extreme}`; metric = SR (AUC).

| Corruption | Severity | Baseline | +Correction | Delta |
|---|---|---:|---:|---:|
| RGB low-light | 0.4 | — | — | — |
| RGB over-exposure | 0.4 | — | — | — |
| TIR saturation | 0.4 | — | — | — |
| TIR crossover | 0.4 | — | — | — |
| Token erasure (random 20%) | — | — | — | — |
| Token erasure (burst) | — | — | — | — |
| Cross-modal misalignment | — | — | — | — |

## 3. Error-correction diagnostics

| Setting | Detection AUROC | Localization recall@k | Recovery delta |
|---|---:|---:|---:|
| 8 parity tokens | — | — | — |
| 16 parity tokens | — | — | — |
| 2 BP iterations | — | — | — |
| 3 BP iterations | — | — | — |

## 4. Ablation

| Variant | SR (AUC) | Notes |
|---|---:|---|
| Baseline: cross-attention fusion | — | no codebook, no syndrome |
| + Target codebook (identity only) | — | `P` removed |
| + Parity tokens | — | `M = 16` |
| + Syndrome | — | check outputs used |
| + Iterative BP | — | full model |
| Full, UEP off | — | uniform node degree |

## 5. Minimal viability experiment (first milestone)

```text
Backbone : OSTrack / ViPT style
Modules  : 8 parity tokens + 1 sparse H matrix + 2 rounds neural BP
Protocol : destroy 20% / 40% of RGB or TIR target tokens
Question : can tracking performance be recovered?
```

| Token corruption | Baseline SR | CodeTrack SR | Recovered |
|---|---:|---:|---:|
| 20% RGB tokens | — | — | — |
| 40% RGB tokens | — | — | — |
| 20% TIR tokens | — | — | — |
| 40% TIR tokens | — | — | — |

**Decision rule**: if recovery is clearly positive at 20% and 40%, the direction is worth scaling.
