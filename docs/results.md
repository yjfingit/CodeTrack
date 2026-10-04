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

### 6.0 Status of the claims (read this first)

One line per claim the project makes or is asked about, with where the measurement lives.  "Not
resolved" means the design cannot separate the effect from noise at the current sample size, not
that the effect is absent.

| claim | measured status | section |
|---|---|---|
| RGB/TIR frame pairing is correct | 220 703/220 703 testingset frames paired (100.000 %; the old rule left **16 257** of them unpaired) and 512 817/514 081 trainingset frames (99.754 %, one sequence refused rather than mixed) | 6.1 |
| the update direction is useful where it is needed | per-token oracle step `a*` is non-zero for **every** damaged token; the error at the *best global* step nearly reaches the oracle | 6.2, 6.9 |
| the update magnitude is wrong | applied/oracle = **1.677** and the update *hurts* (`L1` 0.2926 vs 0.1856 at the zero step); one forced step gives 1.062 and a marginal help | 6.9, 6.10 |
| the learned severity gate carries per-token information | no: permuting it inside (reliability, `||d||`) strata changes the error by <= 0.001 in every arm, and `rho(gate, a*)` is -0.014 (shipped) to +0.096 (D) | 6.9, 6.10 |
| the gate matters at all | **for the level, not the drop**: removing it changes the robustness composite by +0.89 (p = 0.53, `post_norm`) and +0.84 (p = 0.65, identity-residual), but *training* without it lands 4.40 clean points lower (`post_norm`, p = 0.040) and **10.76 points lower** (identity-residual, p = 0.0002) | 6.24 |
| the gate is the applied coefficient | no: the applied coefficient is `(1 - r) * gate`; the old AUC read a selector artefact | 6.4, 6.7 |
| the incidence `H` matters | shuffling it leaves tracking unchanged and destroys localization | 6.3 |
| the syndrome detects damage | no: AUROC **0.345-0.499** (at or below chance) on four corruption families | 6.7.1 |
| the reliability head detects damage | it detects *zeroed tokens*: AUROC 1.0000 on three zeroing families (two held out) and **0.7440** on a held-out non-zeroing family | 6.7.1 |
| a deviation target fixes the shortcut | **yes on the pre-registered endpoint**: 0.7440 -> **0.8123** on the held-out non-zeroing family, and the repair turns from -0.02 to **+0.107** on the training-like mechanism | 6.20 |
| the correction branch repairs damage | **+5.5 %** under the training configuration (both modalities erased, 83 % of sequences positive); -4 % for rgb-only erasure and **-88 %** on a held-out non-zeroing family -- it repairs its own configuration and nothing else | 6.7.1, 6.21 |
| image-level degradation is trained for | **yes**, in the dataset: the search frame gets all four declared corruptions at severity 0.4 (template stays pristine), so the single-corruption evaluation conditions are milder than training, not out of distribution | 6.21 |
| the branch makes the tracker more robust | **not resolved**: difference in differences 0.07-2.41 SR points, all intervals spanning zero, against an MDE of 7.6 at this design | 6.16 |
| the identity-residual parameterisation fixes the update geometry | **yes**: `applied / oracle` 1.015 / 1.004 and `gain vs zero` **+0.0145 / +0.0151** (the update *reduces* the damaged-token error), against 34.4 / 35.0 and -5.79 / -5.91 for the retrained `post_norm` arms | 6.10, 6.24 |
| the parameterisation improves robustness | **not resolved**: the pre-registered A-vs-D composite is **-0.80** SR points ([-5.11, +3.40], p = 0.72), the same null on the mean-IoU endpoint (-0.87 points, p = 0.70); the interval bounds any effect at ~4-5 points | 6.24 |
| the parameterisation improves clean tracking | **yes for the design combination**: identity-residual output *with* the learned gate (arm C) reaches **0.385 clean SR** -- best of the four arms and +4.6 over the shipped checkpoint (A 0.288, B 0.244, D 0.277) | 6.22, 6.24 |
| the parameterisation is free | **no**: the identity-residual arms need the crop clamp on 1.5-1.7 % of frames against 0.00-0.14 % for the `post_norm` arms, and one sequence is the worst case in three arm/condition pairs | 6.24 |
| does a better detector/repairer track better? | **no, it tracks worse**: the deviation-trained arm matches on the pre-registered non-zeroing condition (+1.11 points, p = 0.23) and loses **3.65 SR points** in aggregate (p = 0.013) despite the better held-out AUROC and the positive repair gain | 6.26 |
| can ordinary spatial context explain the results? | **largely yes**: the parameter-matched spatial mixer matches the shipped BP arm (0.353 vs 0.339 clean; 0.252 vs 0.249 on `tok_block_rgb_04`) and is statistically indistinguishable from the best BP arm on the robustness composite (+0.10 points, [-2.50, +2.55]); the BP arm leads by 2.6-3.9 points on the level, only one condition nominally significant | 6.25 |
| TIR token erasure improves tracking | no: the effect disappears when the crops are held fixed (a trajectory effect), and it is identical with the branch switched off | 6.12, 6.16 |
| the extra diagnostics forward biases the comparison | no: +0.3 to +0.5 points, not significant | 6.11 |
| a diverged closed loop is a real failure mode here | yes: 8 928 of 10 244 frames (87.2 %) hit the crop bound under `both@0.4`, the worst sequence 197/200 and five sequences above 190; the counter is now reported with every run | 6.15 |
| what the current design could ever detect | 3 SR points needs ~151 sequences at m = 1; 1 point needs >1300.  The enlarged 150-sequence split is built and ready | 6.13, 6.17 |
| the review round left the default decoder path bit-identical | **retracted**: `073c223` froze the branch input for every non-identity output mode, so `post_norm`/`mlp`/`spatial` changed function from round 2 on (max abs difference 0.08-0.24 at output scale 1).  `model.decoder_state_mode` now names the semantics and the guard refuses a mismatch | 6.27.1 |
| `applied/oracle = 34.4` is a step-size overshoot | **misnamed**: it is `L1(applied)/L1(oracle)`, an error ratio.  The step ratio `a_applied/a*` is now reported separately | 6.27.3 |
| the fixed-crop replay saw the same crops as the free run | **no**: the reference schedule was applied one frame early, and the check that was supposed to catch it was `... or len(gt) > 0`, i.e. always true.  Fixed and to be re-run; the TIR reading of 6.12 is provisional | 6.12, 6.27.3 |
| "N frames clamped" means the loop diverged | **too coarse**: the counter mixed scale, centre, non-positive and non-finite events.  Only the scale count means unbounded growth; the breakdown is now stored per run and per sequence | 6.15, 6.27.3 |
| the family-wise error rate of 6.13 was the family-wise rate | **no**: it counted P(this hypothesis rejected) and drew an independent sample per comparison.  Corrected to P(any rejection) with shared resampling; 6.13's power figures are superseded | 6.13, 6.27.3 |
| the 6.26 composite is pre-registered | **no**: 6.20 pre-registered `tok_noise_rgb_04` (primary) and `tok_block_rgb_04` (secondary); the two-condition composite is a post-hoc aggregation and is now marked `in_holm_family: false` and exploratory | 6.20, 6.26, 6.27.3 |
| every number here can be tied to the code that produced it | **no**: the provenance schema is new, so all 74 pre-existing runs are unattributable by construction and the round-2 numbers must be produced again rather than extended | 6.27.2 |


Two structural notes that apply to every row: all paired intervals resample **sequences** (never
frames), and every family of tests is corrected with Holm over *the whole family*, condition
comparisons and interactions together.  The `--composite` output of `tools/summarize_paired_conditions.py`
is **outside** that family by construction (it is defined by the conditions named on the command
line) and is stored with `in_holm_family: false`; it is exploratory unless a pre-registration names
the aggregation rule in advance (section 6.27.4).

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
  for. It holds **for the pre-norm cell only** -- the shipped output is `LN(v)`, which is not
  the input, so "gate 0 leaves the token untouched" is a statement about the residual path,
  not about the model's output.
* Every damaged-token gain has **two denominators** and both must be quoted:
  `(0.1968 - 0.1919) / 0.1968 = 2.49 %` answers "is the overall output closer to clean than
  the corrupted input", while `(0.2274 - 0.1919) / 0.2274 = 15.6 %` answers "how much do the
  messages add on top of what the output LayerNorm already does". They are not
  interchangeable, and a single "gain" column silently picks one of them.
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

### 6.5 Stratified long-horizon validation (complete for `ab_full`)

60 `testingset` sequences x up to 200 frames, 12 per attribute stratum
(`outputs/validation_split_v1/`), `outputs/ab_full/final.pth`
(`outputs/validation_v1/summary_ab_full.json`). All nine conditions of the main matrix have now
finished; the no-decoder and MLP arms and the ratio-0 control are separate runs.

Clean baseline (`outputs/validation_v1/ab_full/clean/metrics.json`):

| stratum | n | clean SR (AUC) | median |
|---|---:|---:|---:|
| unoccluded | 12 | 0.478 | 0.496 |
| partial occlusion | 12 | 0.343 | 0.359 |
| total occlusion | 12 | 0.330 | 0.189 |
| thermal crossover | 12 | 0.300 | 0.240 |
| low illumination | 12 | 0.246 | 0.112 |
| **all** | **60** | **0.340** | 0.291 |

The same checkpoint scores 0.72 SR on the 20-sequence / 12-frame convenience sample the earlier
curves used, so the two are not comparable: the stratified split is harder *and* 200 frames is a
much longer horizon. `tools/horizon_probe.py` reports the mechanism (loss-time distribution and
per-frame IoU trace). A three-sequence pilot showed the typical failure is not slow drift -- on
`basketballathand` IoU falls 0.78 -> 0.23 within seven frames while the target moves ~5 px/frame,
while `mandownstair` (slow motion) tracks at 0.80 for all 200 frames.

Paired clean-vs-corrupt deltas, 60 sequences, positive = corruption hurt. `Holm p` is the
Holm-Bonferroni value over the whole family of eight condition tests (the family now also takes
in any `--interaction` comparisons, see 6.10):

| condition | clean SR | corrupt SR | drop (pts) | 95% CI (pts) | rel % (per-seq) | rel % (pooled) | Holm p |
|---|---:|---:|---:|---|---:|---:|---:|
| `rgb_lowlight_04` | 0.3395 | 0.3293 | +1.02 | [-0.96, 3.10] | -1.99 | 3.01 | 1.0000 |
| `rgb_occl_04` | 0.3395 | 0.2296 | **+10.99** | [6.65, 15.79] | 22.17 | 32.36 | **0.0001** |
| `tir_crossover_04` | 0.3395 | 0.3328 | +0.67 | [-0.51, 1.96] | -1.39 | 1.96 | 1.0000 |
| `tok_block_both_02` | 0.3395 | 0.3531 | -1.36 | [-4.45, 1.63] | -25.80 | -4.01 | 1.0000 |
| `tok_block_both_04` | 0.3395 | 0.0370 | +30.25 | [24.21, 36.64] | 72.48 | 89.10 | 0.0000 |
| `tok_block_rgb_02` | 0.3395 | 0.3051 | +3.44 | [0.99, 6.34] | 7.88 | 10.13 | 0.0901 |
| `tok_block_rgb_04` | 0.3395 | 0.2492 | **+9.03** | [5.03, 13.54] | 21.64 | 26.60 | **0.0009** |
| `tok_block_tir_02` | 0.3395 | 0.3868 | -4.74 | [-9.20, -0.41] | -54.32 | -13.95 | 0.1984 |
| `tok_random_rgb_02` | 0.3395 | 0.3347 | +0.47 | [-0.80, 1.92] | 1.49 | 1.40 | 1.0000 |

Reference-tracking subset (clean SR > 0.2, fixed by the clean arm, n = 32):

| condition | clean SR | corrupt SR | drop (pts) | 95% CI (pts) | p(t) |
|---|---:|---:|---:|---|---:|
| `rgb_occl_04` | 0.5611 | 0.3627 | **+19.84** | [12.74, 27.35] | 0.0000 |
| `tok_block_rgb_04` | 0.5611 | 0.4065 | **+15.45** | [8.41, 23.01] | 0.0003 |
| `tok_block_rgb_02` | 0.5611 | 0.5003 | +6.08 | [1.60, 11.15] | 0.0193 |
| `tok_block_both_02` | 0.5611 | 0.5338 | +2.73 | [-1.39, 6.87] | 0.2076 |
| `rgb_lowlight_04` | 0.5611 | 0.5342 | +2.69 | [-0.12, 5.98] | 0.0994 |
| `tir_crossover_04` | 0.5611 | 0.5427 | +1.83 | [0.18, 3.96] | 0.0739 |
| `tok_random_rgb_02` | 0.5611 | 0.5531 | +0.80 | [-1.50, 3.46] | 0.5386 |
| `tok_block_tir_02` | 0.5611 | 0.5799 | -1.88 | [-7.96, 4.67] | 0.5693 |

Pre-registered strength screen (20-40 % relative drop, 60-80 % retained SR, interval excludes
zero). The pre-registration is defined on the frozen *no-decoder* arm; this is the same
computation on the decoder arm as a supplemental calibration:

| condition | rel % (per-seq) | rel % (pooled) | retained % | interval excludes 0 | verdict |
|---|---:|---:|---:|---|---|
| `rgb_occl_04` | 22.17 | 32.36 | 67.64 | yes | **pass** |
| `tok_block_rgb_04` | 21.64 | 26.60 | 73.40 | yes | **pass** |
| `tok_block_rgb_02` | 7.88 | 10.13 | 89.87 | yes | too weak (and n.s. after Holm) |
| `tok_random_rgb_02` | 1.49 | 1.40 | 98.60 | no | n.s. |
| `rgb_lowlight_04`, `tir_crossover_04` | -2.0 / -1.4 | 3.0 / 2.0 | 97.0 / 98.0 | no | n.s. (no effect at severity 0.4) |
| `tok_block_tir_02` | -54.32 | -13.95 | 113.95 | yes (raw) | not significant after Holm (0.198) |
| `tok_block_both_02` | -25.80 | -4.01 | 104.01 | no | n.s. |

Per-stratum paired drop (points, positive = corruption hurt):

| condition | low illum | partial occl | thermal cross | total occl | unoccluded |
|---|---:|---:|---:|---:|---:|
| `rgb_occl_04` | +7.83 | +8.36 | +7.93 | +8.59 | **+22.23** |
| `tok_block_rgb_04` | +10.24 | +7.62 | +8.38 | +2.92 | +16.00 |
| `tok_block_rgb_02` | +3.31 | +4.19 | +0.49 | +2.22 | +6.98 |
| `tok_random_rgb_02` | +2.22 | -0.36 | -0.05 | +0.50 | +0.06 |
| `rgb_lowlight_04` | -0.46 | +2.89 | +1.50 | +0.98 | +0.21 |
| `tir_crossover_04` | -1.51 | +1.29 | +2.15 | +0.06 | +1.33 |
| `tok_block_both_02` | -4.42 | -1.34 | +4.21 | -5.57 | +0.32 |
| `tok_block_tir_02` | +1.69 | -1.39 | -3.77 | -9.10 | -11.10 |

`tok_block_both_04` is in the table for completeness but **is not a degradation measurement**: the
tracker's closed loop collapses under it (see 6.15 -- 87.15 % of its frames needed the box clamp,
and its worst sequences spend 190-197 of 200 frames clamped).  Its "drop" describes a dead loop,
which is why the strength screen labels it `too strong` and why it is excluded from the
calibrated set below.

What the completed matrix says:

* **The calibrated degradations are `rgb_occl_04` (+11.0 pts overall, +19.8 on the tracked
  subset, Holm p = 0.0001) and `tok_block_rgb_04` (+9.0 pts, Holm p = 0.0009).** Everything
  else is either too weak or not significant once multiplicity is accounted for.
* The **image-level RGB occlusion is the strongest** condition -- and it is exactly the class of
  condition that silently scored zero sequences before the P0 fix, so it had never been measured
  at all.  RGB low-light and TIR crossover at severity 0.4 move SR by ~1 point (n.s.): the image
  corruptions as configured are not a degradation for this tracker.
* **RGB token block erasure at ratio 0.2 is not significant after Holm** (0.090); at 0.4 it is.
  Report the curve, not the level.
* **The TIR "improvement" does not survive correction** (raw p 0.040 -> Holm 0.198), and
  section 6.12 shows it should not be read as a benefit: with the crops fixed to the clean arm's
  trajectory the effect reverses sign (-0.0017 mean IoU, -0.0194 single-step).  It is closed-loop
  dynamics, not a per-frame improvement.
* Block versus random erasure at the *same* ratio (0.2) is +3.44 pts against +0.47 pts: the
  spatial structure of the damage matters, which is the premise the locality prior was built on,
  though neither condition survives correction on its own.
* These are not population estimates: 60 stratified sequences, paired intervals over sequences,
  one checkpoint, one severity per condition, one 200-frame horizon.
* Commands:
  `python tools/select_validation_sequences.py --out outputs/validation_split_v1 --n 60 --seed 0`;
  `bash tools/validate_sequences.sh` (see `docs/corruption_protocol.md` section 6);
  `python tools/summarize_paired_conditions.py --run ab_full=outputs/validation_v1/ab_full --reference clean --strata-manifest outputs/validation_split_v1/manifest.json --screen-run ab_full --out outputs/validation_v1/summary_ab_full.json`.

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
| `1 - reliability_rgb` | **1.000** | the reliability head ranks the injected damage (nearly) perfectly on this distribution; it is trained with `BCE(r, 1 - mask)` |
| `locator_scattered` (`H^T s`) | 0.853 | the locator ranks damaged tokens well; also mask-supervised |
| `selection_of_damage` | 0.812 | the selector preferentially puts damaged tokens into the graph |
| `gate_within_selected` | 0.392 | within the selected nodes, the learned severity gate is lower on damaged tokens than on healthy ones |
| `effective_gate_rgb` (unselected keep 1) | 0.147 | the severity gate scattered over the grid; **not** the applied coefficient -- the selector's preference for damaged tokens makes the neutral 1s concentrate on healthy tokens, so this number cannot be read as "the decoder suppresses the repair" |
| `applied_coefficient` = `(1 - r) * gate` | *added by the P0 fix; re-run required* | what the decoder actually multiplies the update by |
| `syndrome_density_rho` (within frame) | 0.764 | the syndrome tracks each check's realized corruption density |

**Correction (imported from the external review, verified against the code).** The first
version of this section claimed the decoder "suppresses the correction where it is needed"
from `effective_gate_rgb = 0.147`. That is not what the number shows: `tools/gate_information.py`
scored the severity gate **without** multiplying by `(1 - r)`, and `gate_information` mixes
selected values with the neutral 1s of unselected variables. The applied coefficient is
`(1 - r) * gate`, `(1 - r)` is ~1 on injected damage and ~0 on healthy tokens, so the sign of
the effect is not established by that AUC. The corrected score is now computed as
`applied_coefficient` and is pending a re-run.

What the table still supports, with the right wording:

1. The heads rank the *injected* damage well. That is "predicts the injected mask
   distribution almost perfectly"; whether it is memorisation or a trivially generalisable
   cue (zero erasure is an easy signal) can only be decided on a **held-out corruption
   family**, which has not been run.
2. `(1 - r)` already carries the damage information by itself, so a second learned severity
   gate has to justify what it adds beyond `(1 - r)`; whether it does is exactly what the
   P1 step-size/gate-utility diagnostics are for, and it is not settled here.
3. Forcing the gate open (`full`/`oracle`, section 6.4) drives the *pre-norm* error from 0.33
   to 0.79 while the post-norm error barely moves. That is consistent with "the update
   overshoots and the output LayerNorm absorbs the scale", but it does not prove that the
   suppression is why the gate learned its level.

Two consequences that survive the correction:

* Detection/localization AUROC is not evidence of error-correction capability -- those heads
  are supervised with the synthetic mask. Report them as "predicts the injected mask" and pair
  them with a mask-independent check.
* Any claim that the Tanner structure improves tracking has to come from the paired
  difference-in-differences on held-out sequences (`--interaction`), not from these AUCs.

#### 6.7.1 Held-out corruption families: what the AUROC 1.000 actually is

(The measured status of these claims is mirrored into `docs/architecture.md` next to the metric
definitions, so a reader of the architecture never sees the stronger pre-measurement wording
without it.)

The objection to the reliability head's AUROC 1.000 was that it may simply memorise the training
corruption function.  The checkpoint was trained with exactly one family, `tok_burst_erase`
(contiguous run of zeroed tokens), so three other families are genuinely held out.  Same 8
sequences x 40 frames, RGB only, ratio 0.2, severity 0.4, `--diagnostics`
(`outputs/validation_v1/family_*/metrics.json`; the reliability AUROC is now recorded by the
evaluation itself, in `codetrack/engine/evaluator.py::summarize_detection`, because scoring that
head only in `tools/gate_information.py` left the *training* family unscored):

| corruption family | mechanism | reliability AUROC (1 - r) | syndrome AUROC | locator precision@5 |
|---|---|---:|---:|---:|
| `tok_burst_erase` (seen) | tokens zeroed | 1.0000 | 0.373 | 0.838 |
| `tok_block_erase` (held out) | tokens zeroed | **1.0000** | 0.392 | 0.783 |
| `tok_random_erase` (held out) | tokens zeroed | **1.0000** | 0.345 | 0.312 |
| `tok_feat_noise` (held out) | noise added, nothing zeroed | **0.7440** | 0.499 | 0.212 |

Three conclusions, each of which changes how the earlier numbers must be quoted:

* **The 1.000 is a mechanism shortcut, not family memorisation.**  It survives two held-out
  families with the same *mechanism* (zeroing) and drops to 0.744 as soon as the mechanism
  changes.  So the head detects "this token was zeroed", which generalises over mask geometry
  but not over corruption type.  The honest single number for "detects damage" is **0.744**, and
  any claim of degradation detection should be scored on a non-zeroing family.
* **The syndrome does not detect damage at all** (AUROC 0.345-0.499, i.e. at or below chance,
  across every family).  Whatever the parity/syndrome path contributes, it is not "this token is
  damaged"; this is the same picture as section 6.3, where shuffling the incidence left tracking
  unchanged and destroyed localization.
* **Localization exploits contiguity**: precision@5 is 0.84/0.78 for contiguous damage (burst,
  block) but 0.31/0.21 for scattered damage (random, noise) against a chance level of ~0.2.
  The "locator" is a compact-region finder, which is exactly what the locality prior in `H`
  builds in.

**The repair side of the same runs is worse than the detection side.**  The recovery diagnostics
of those identical evaluations (`recovery_gain`, per-token L1 deviation from the frozen teacher
on the corrupted positions, recorded in the same `metrics.json`):

| corruption family | error before | error after | ratio | `recovery_gain` |
|---|---:|---:|---:|---:|
| `tok_burst_erase` (seen, zeroing) | 0.2115 | 0.2157 | 1.02x | -0.0196 |
| `tok_block_erase` (held out, zeroing) | 0.2190 | 0.2236 | 1.02x | -0.0209 |
| `tok_random_erase` (held out, zeroing) | 0.2075 | 0.2114 | 1.02x | -0.0185 |
| `tok_feat_noise` (held out, **non-zeroing**) | 0.1229 | 0.2304 | **1.88x** | **-0.8754** |

* On the training-like mechanism the decoder makes the tokens ~2 % **worse**, i.e. it is a
  near-no-op that does not repair.
* On the held-out non-zeroing mechanism it **nearly doubles** the error (1.88x).  The corruption
  there is milder to begin with (0.123 vs 0.211), so in relative terms the update is far more
  damaging: the branch applies a near-constant "repair" tuned to zeroed tokens, and when the
  damage has a different magnitude and direction the same update adds error instead of removing
  it.

Together with the AUROC rows this is the sharpest statement the current evidence supports about
the correction branch: it is a **corruption-mechanism-specific operator**, and neither its
detection nor its repair transfers to a mechanism it was not trained on.

**Correction (added in section 6.21).**  The table above was measured with `--corrupt-target rgb`,
while training erases **both** modalities.  With the training setting the same checkpoint reports
`recovery_gain` **+0.055** instead of -0.042, so the "near no-op / inflates the error" reading
applies to the *mismatched* configuration, not to the trained one.  What survives is the
specificity: the branch repairs the configuration it was trained on and not rgb-only erasure, not
a non-zeroing family, and not image-level degradation.

The P4 supervision change (`loss.detect_target="deviation"`, a continuous feature-deviation
target) is the designed answer to the first point, and this table is its baseline: a head
trained against the mask reaches 0.744 on the non-zeroing family, which is the number the
deviation-trained head had to beat -- it reached **0.8123** (section 6.20); the P2 bounded-step parameterisation is the answer to the
second, because a clipped identity-residual step cannot add 87 % error by construction.  The
pre-registration for the P4 run is section 6.18.

### 6.8 Strict no-op control (`--corrupt-identity`)

The corrupted conditions run an extra clean-reference forward, and `evaluate()` used to make
that pass conditional on `corruption is not None`, so "clean" and "corrupt" differed in more
than the corruption. `--corrupt-identity` is a strict no-op: diagnostics enabled, no
image-level degradation, no token mask, and **no RNG draw** (unlike `--corrupt-ratio 0`, which
`corrupt_tokens` clamps to one erased token per frame). `tools/validate_sequences.sh` exposes
it as the `noop_diag` condition; the acceptance check is that `noop_diag` and `clean` produce
identical per-sequence metrics (verified: 4 sequences x 20 frames, per-sequence SR/PR/NPR are
byte-identical, and the identity run logs `identity corruption` for every sequence).

### 6.9 Step size vs gate utility: which explanation survives (`tools/step_size_probe.py`)

48 frames / 2448 damaged tokens, RGB block erasure at ratio 0.2, `ab_full`
(`outputs/step_size_probe_ab_full.json`, `outputs/step_size_probe_ab_full_1step.json`).
`d` is the update with the gate forced open, `a* = clip(<d, e> / ||d||^2, 0, 2)` is the
**offline oracle** step (`e = clean - v`), `applied` is what the shipped model does, and
`gate-permuted` re-runs the same step with the learned gate shuffled *within*
(reliability, `||d||`) strata, so only its value information is destroyed.

| configuration | L1 zero | L1 applied | L1 oracle | L1 best global step | L1 gate permuted |
|---|---:|---:|---:|---:|---:|
| 2 iterations (as shipped) | 0.1856 | **0.2926** | 0.1745 | 0.1753 | 0.2936 |
| 1 iteration (`--single-step`) | 0.1856 | 0.1838 | 0.1730 | 0.1731 | 0.1838 |

* `a*` is **never** zero (fraction 0.000 at both settings): for every damaged token in this
  sample there is a positive step that lowers the error. The failure is therefore **not** "the
  direction is useless on many tokens" -- selective rejection is not the explanation and the
  uniform-step reading survives.
* A **single global step closes essentially the whole oracle gap** (0.1753 vs 0.1745 at two
  iterations; 0.1731 vs 0.1730 at one). One scalar, not a per-token policy, is what the
  current update needs.
* At the shipped two iterations the aggregate update makes damaged tokens **worse than doing
  nothing** (0.2926 vs 0.1856), while one iteration slightly improves them (0.1838): the
  overshoot accumulates across iterations.
* The learned severity gate is **decoration**: shuffling its values within the covariates that
  could explain it changes the L1 error only from 0.2926 to 0.2936 (two iterations) and leaves
  it at 0.1838 (one step); it correlates with `a*` at rho -0.014 / -0.024; its median on
  damaged tokens is 0.004 (mean 0.179, and every damaged token here was selected). It nearly
  closes the update and carries no per-token information beyond `(1 - r)` and `||d||`.

Consequence for the P2 four-arm grid: the candidate worth testing is *identity residual output
+ bounded (or global) step with the severity gate forced to one* (arm D), not "keep the gate
and rescale it". The oracle step is an offline diagnostic and must never be reported as a
deployment result.

### 6.10 P2 design: identity-residual output, bounded step, and the severity-gate question

Section 6.9 says the update direction is usable everywhere and that one global step closes the
oracle gap, while the learned severity gate carries no per-token information.  The four-arm
grid that tests those two conclusions is:

| arm | output parameterisation | severity gate | what it isolates |
|---|---|---|---|
| A | `post_norm` (shipped) | learned | reproduction anchor |
| B | `post_norm` | forced to 1 | does the shipped parameterisation depend on gate suppression |
| C | `identity_residual` | learned | is the second gate worth anything once the scale is sane |
| D | `identity_residual` | forced to 1 | the candidate: identity path + bounded step, no learned gate |

All four share the recipe and budget of the existing arms (6000 steps, `codec_warmup_epochs=0`,
one seed, `bash tools/validate_sequences.sh` on the same 60-sequence split), and all four are
trained from the same seeded initialisation for the common modules.  Arms C/D also use
`model.residual_clip=19.045`, the p99 of the ideal residual `||x* - x||_2` measured on a
pre-declared 128-frame calibration split (`outputs/residual_budget.json`,
`tools/calibrate_residual_budget.py`); the bound is an engineering budget, not a value tuned on
the evaluation set.  A direct rewrite of the old `ab_full` checkpoint with the output LayerNorm
removed is explicitly **not** part of this grid: that would measure the old head's reaction to a
changed input distribution rather than the new parameterisation.

Training artifacts land in `outputs/p2_{A,B,C,D}/` with a `resolved_config.json` each, the
per-arm step-size readout in `outputs/step_size_probe_p2_*.json`, and the tracking runs in
`outputs/validation_v1/p2_*/`.

**Step-1 readout, all four arms (corrected -- see the note below).**
`tools/compare_step_probes.py` over the arm artifacts, on the same 48 frames and matched masks
(`outputs/step_probe_table.json`); the shipped checkpoint is the historical reference of the same
parameterisation:

| arm | `applied / oracle` | L1 at zero step | L1 applied | L1 at oracle step | gain vs zero | gate permutation | `a* = 0` share |
|---|---:|---:|---:|---:|---:|---:|---:|
| shipped `ab_full` (`post_norm`) | 1.677 | 0.1856 | 0.2926 | 0.1745 | -0.1070 | 0.0010 | 0.000 |
| A (`post_norm`, learned gate) | **34.39** | 0.1856 | **5.9792** | 0.1739 | **-5.7936** | -0.0041 | 0.000 |
| B (`post_norm`, gate = 1) | **35.02** | 0.1856 | **6.0908** | 0.1739 | **-5.9052** | -1.1021 | 0.000 |
| **C (`identity_residual` + clip, learned gate)** | **1.015** | 0.1856 | **0.1711** | 0.1686 | **+0.0145** | 0.0000 | 0.000 |
| **D (`identity_residual` + clip, gate = 1)** | **1.004** | 0.1856 | **0.1705** | 0.1699 | **+0.0151** | 0.0013 | 0.000 |
| P5 spatial mixer (`decoder_mode=spatial`, `post_norm`) | 2.796 | 0.1856 | 0.4862 | 0.1739 | -0.3006 | 0.0014 | 0.000 |

The spatial-mixer control (P5, section 6.14) belongs in the same table: it keeps the default
`post_norm` output, and its update overshoots by 2.80x with `gain vs zero` -0.30 -- i.e. between the
two extremes, much milder than the BP `post_norm` arms (34-35x, -5.8) and much worse than the
identity-residual arms (1.00-1.02x, *positive* gain).  Its learned gate also sits wide open
(median 0.992), which is what a decoder looks like when its step is only moderately too large.

Read the last column first: `gain vs zero` is **positive for C and D** and negative for the shipped
and `post_norm` arms.  Positive means the update *reduces* the per-token error of the damaged
tokens: D takes 0.1856 to **0.1705** (-8.1 %) with `applied / oracle = 1.004` -- i.e. at essentially
the ideal step size, and within 0.0006 of what any global scale of its own direction could achieve
(`best global` 0.1700).  C is the same picture with the learned gate left in place (0.1711,
`applied / oracle = 1.015`), and its gate opens to a median of **0.840** instead of the shipped
arm's 0.004 -- with a bounded, correctly-scaled step there is nothing left for the gate to
suppress, and `rho(gate, a*)` turns positive (+0.053 / +0.096 against the shipped -0.014).

Conclusions, all visible without a single tracking run:

* **The bounded identity-residual parameterisation fixes the scale *and* makes the update repair.**
  That is the P2 design hypothesis, confirmed at the geometry level: `post_norm` arms overshoot by
  34-35x and push the damaged-token error up by 5.8-5.9 L1, while the identity-residual arms sit at
  1.00-1.02x and *reduce* it.
* **`post_norm` is not reproducible in outcome**: two runs of the same configuration land 20x apart
  in scale (shipped 1.68x vs A 34.4x), because the output LayerNorm absorbs the update's magnitude.
* **The gate is inert in every arm** (permutation effect 0.000-0.001 for C, D and the shipped arm;
  note B's -1.10 is the gate-forced arm where permutation moves the *unforced* gate), so 6.9's
  conclusion survives on retrained arms as well.

**The earlier version of this table was wrong for B, C and D, and the new load guard is how it was
found.**  The probes had been run without each arm's runtime overrides, so C and D were probed
through a `post_norm` decoder that silently dropped their `branch_norm`/`residual_out`, and B was
probed without `gate_always_one`.  That produced a table in which D "achieved do-no-harm rather
than repair, with a useless direction" (`a* = 0` for 43 % of tokens, applied L1 0.1976) -- which is
the *opposite* of the truth.  After extending `load_checkpoint` to refuse those mismatches
(6.23), each arm was re-probed with its overrides; the invalid artifacts are kept as
`outputs/step_size_probe_p2_{B,C,D}_invalid_arch.json` and the corrected ones are canonical.  The
lesson is the same as 6.23's: an output that looks plausible is not evidence, and a guard that
refuses a mismatched load is worth more than a warning nobody reads.

**Incident: arm C aborted under AMP, arm D did not.**  Arm C crashed with a CUDA device-side
assert (`Loss.cu: Assertion input_val >= zero && input_val <= one failed`) about 2 200 steps in,
i.e. `BCELoss` received a NaN probability.  Arm D -- the same parameterisation with the gate
forced to 1 -- trained all 6 000 steps without incident, so the failure is not inherent to the
identity-residual path.  The decoder now replaces a non-finite update with zero and counts it
(`decoder_nonfinite_steps`, logged as a warning, covered by a unit test), which makes such an
event diagnosable instead of fatal, and C is being retrained in fp32 with that guard in place
(`outputs/p2_C_fp32.log`).  If the fp32 run also fails, that is a numerical-stability finding
about the un-normalised output rather than something to hide.

**Provenance: the grid is internally comparable, but arm A is not a bit-identical replay of the
shipped `ab_full`.**  The four arms were trained in one session with the same code, seed, recipe,
budget and `--override` set, so differences *between them* are attributable to the arm settings.
The shipped checkpoint predates the review-round code changes, and its final training losses
differ slightly from arm A's (loss 3.2492 vs 3.1390; `track` 0.4205 vs 0.4826; `detect` 1.7573 vs
1.6508).

**Correction (6.27.1).**  This paragraph first attributed that difference to "the corruption/sampling
RNG stream having shifted at some point between the two runs".  That was a guess, and it was not
checked against the one difference that is *guaranteed* to change the loss: `073c223` silently
changed the decoder loop's state semantics, so from round 2 on the message functions read the
decoder's input instead of the state round 1 produced.  The two candidate causes are not
distinguishable from the loss values, and the honest status is **not attributed**.  Arm A is a
`held`-semantics arm and needs `--override model.decoder_state_mode=held` to be rebuilt; the
default is now `recurrent`, which is what `ab_full` was trained under.  The equivalence run is
`outputs/decoder_state_equivalence.json`.

**Metric naming (6.27.3).**  The `applied / oracle` values quoted below are
`L1(applied)/L1(oracle)`, an **error** ratio; they are not the step-size overshoot factor.  The
step ratio `a_applied/a*` is reported separately by `tools/step_size_probe.py`, because the two
numbers were previously conflated and 34.4 was read as "the step is 34 times too large".

Therefore: **no result is reported as A vs `ab_full`**; the shipped arm is quoted only as the
historical reference whose step-probe ratios the new arms are compared against.

The P4 arm's losses are a useful cross-check of the deviation target: `detect` 1.0510 with
`rel` 0.276 and `syn` 0.358, against 1.6508 / 0.014 / 0.568 for the mask target -- the soft target
has a higher BCE floor by construction (section 6.18) but the syndrome fits its (now continuous)
target far better, which is consistent with the improved syndrome AUROC in section 6.20.

**Pre-registered analysis recipe** (fixed before the runs finish, so the grid cannot be read
selectively afterwards).  Its outcome is section **6.24**; the spatial control that section 6.10
calls step 0's comparison is section **6.25**.

0. **Secondary diagnostic, not an endpoint**: `tools/gradient_conflict_report.py` reads the
   `[grad] cos(L_track, L_correct)` series every arm already logs, so "this arm tracks worse" can
   be separated from "the two losses fight in this arm".  Measured on the runs available so far:
   shipped `ab_full` mean +0.063 (min -0.250), `p2_A` +0.093 (min -0.119), `p4_deviation` +0.128
   (min -0.073) -- mildly aligned in all three, with the correction gradient 0.17-0.23x the
   tracking gradient, i.e. no sign of the conflict the project once worried about.
The whole recipe is one command, `bash tools/p2_grid_report.sh`, which runs steps 0-6 below,
writes every table to `outputs/p2_report/`, and is meant to be run *after* the arm runs exist (it
waits on nothing).  Its steps:

1. **Update geometry first, tracking second.**  `tools/compare_step_probes.py` over the four arms.
   An arm counts as "step size fixed" when `applied / oracle` is near 1 and `L1(applied) <
   L1(zero)`; the severity gate "earns its place" only if permuting it inside (reliability,
   `||d||`) strata moves the error materially.  The shipped arm's reference numbers are in the
   table below.  A tracking result is only interpreted after this step, so a null can be
   attributed to a mechanism.
2. **Primary contrast, m = 1**: **A vs D**, the whole parameterisation change (identity-residual
   output + bounded step + no learned gate), reported as one `--composite` over the two token
   conditions (`tok_block_rgb_04`, `tok_noise_rgb_04`) with the same command shape as 6.19:
   `--composite A D --composite-conditions tok_block_rgb_04,tok_noise_rgb_04`.  Weights are 12 %
   *against* D (its decoder has 0.592 M more parameters), which is stated rather than corrected.
3. **Secondary, m = 1 each**: **A vs B** (removing the gate alone, identical parameters) and
   **C vs D** (same parameterisation, gate on or off).  These two are the clean single-factor
   comparisons; everything else is exploratory and joins the Holm family.
4. **The non-zeroing condition is the decisive one.**  Section 6.7.1 measured the shipped decoder
   *inflating* per-token error by 88 % on `tok_feat_noise` while being a ~2 % no-op on erasing
   families, and a clipped identity-residual step is supposed to be unable to do that.  The
   A-vs-D composite therefore gets reported split by condition as well as pooled: a gain that
   appears only on the erasing condition would not support the bounded-step story.
5. **Level check**: `clean` must not differ between arms (a parameterisation that changes
   in-distribution tracking is changing something else), reported as the `--compare` level table.
6. **Divergence is not a metric, it is a validity flag**: `tools/divergence_report.py` for every
   arm, and any arm whose clamped frames are concentrated in a few sequences is reported as such.
7. **Endpoint**: SR is primary; the same composite is repeated with `--metric iou_mean`, which the
   arm runs record (6.17.1 says it is not expected to be less noisy, so it is a robustness check,
   not a tie-breaker).

**Arm construction, verified before the runs** (the decoder alone, reference size; the config
plumbing was resolved the same way for each arm's `--override` list):

| arm | decoder parameters | `branch_norm` | `residual_out` | `residual_clip` |
|---|---:|---|---|---:|
| A `post_norm`, learned gate | 4.9277 M | no | no | 0 |
| B `post_norm`, gate = 1 | 4.9277 M | no | no | 0 |
| C `identity_residual` + clip, learned gate | 5.5199 M | yes | yes (exactly zero-initialised) | 19.045 |
| D `identity_residual` + clip, gate = 1 | 5.5199 M | yes | yes (exactly zero-initialised) | 19.045 |

The two identity-residual arms carry **0.592 M more decoder parameters** (+12 %), because the
parameterisation needs its own branch norm and output projection (1.5 k + 590.6 k).  A difference
between A and C therefore mixes the parameterisation with that capacity, and the honest way to
read it is "the parameterisation, at 12 % more decoder capacity" -- or to compare C/D against each
other, where the capacity is identical and only the gate differs.

**Pre-registered readout for the grid** (`tools/compare_step_probes.py`, which puts several
probe artifacts in one table).  The reference arm's numbers are already measured and fix the
targets:

| arm | applied / oracle | L1(applied) vs L1(zero) | gate permutation effect |
|---|---:|---:|---:|
| `ab_full`, shipped 2-iteration decoder | 1.677 | **-0.107** (the update hurts) | 0.001 (inert) |
| `ab_full`, forced single step | 1.062 | +0.002 (marginal help) | 0.000 (inert) |

The parameterisation is judged first on those ratios, because they are measured without any
tracking noise: an arm is "step-size fixed" when `applied / oracle` is near 1 and
`L1(applied) < L1(zero)`, and the severity gate "earns its place" only if permuting it inside
(reliability, `||d||`) strata changes the error materially.  Tracking results are then read
against that, so a null tracking result can be attributed to a mechanism instead of being
reported as a mystery.  Ordered in the same spirit: `applied/oracle` -> `gain_vs_zero` ->
tracking, and only then the difference in differences of section 6.16.

### 6.11 Ratio-0 control: the extra diagnostics forward does not move the result

The corrupted conditions run an extra clean-reference forward (`collect=True`), so "clean vs
corrupt" differed in more than the corruption.  `--corrupt-ratio 0` still erases one token per
frame, so it is not a strict no-op; it is a *path* control: same code path, diagnostics on,
corruption essentially absent.  Paired against the same 60 sequences
(`outputs/validation_v1/control_ratio0_summary.json`):

| run | clean SR | ratio-0 SR | delta (pts) | 95% CI (pts) | p(t) |
|---|---:|---:|---:|---|---:|
| `ratio0_tir` | 0.3395 | 0.3449 | +0.54 | [0.04, 1.36] | 0.138 |
| `ratio0_rgb` | 0.3395 | 0.3426 | +0.32 | [-0.19, 1.14] | 0.400 |
| `ratio0_both` | 0.3395 | 0.3440 | +0.45 | [-0.13, 1.33] | 0.250 |

All three sit within half a point of clean and none is significant, while the same runner on
the same sequences produced +3.44 pts for RGB block erase at 0.2 and +9.03 pts at 0.4.  Combined
with the strict no-op of section 6.8 (bit-identical per-sequence metrics), the diagnostics pass
is **not** the mechanism behind any corruption effect -- including the TIR sign, which therefore
stands as an unexplained trend (Holm p = 0.198) rather than an artifact.  A path intervention
that holds the crops fixed (P3) is what can separate "TIR content hurts" from "the closed loop
diverged", and that is not measured yet.

### 6.12 Fixed-crop replay and path interventions: the TIR sign is closed-loop dynamics

`tools/trajectory_replay.py` runs a condition three ways on the same frames: normally (its own
predictions drive the next crop), replayed with the **clean arm's per-frame boxes** driving every
crop (all conditions share the history, so a per-frame difference is the model's response), and
with one path replaced by the clean run's tensors.  8 sequences x 40 frames,
`tok_block_erase`, target `tir`, ratio 0.2 (`outputs/trajectory_replay_ab_full_tir_stage2.json`):

| arm | mean IoU over 40 frames | mean IoU, first 5 frames |
|---|---:|---:|
| clean | 0.2823 | 0.7312 |
| corrupt, free crops | 0.2869 | 0.7159 |
| corrupt, crops fixed to clean | **0.2806** | **0.7118** |
| `feature`: clean TIR tokens/taps into the fusion, corrupt control | 0.2874 | 0.7258 |
| `control`: clean reliability + severity gates, corrupt features | **0.2778** | **0.7100** |
| `both` | 0.2874 | 0.7256 |

* With the crops held fixed the corruption **costs** 0.0017 mean IoU and 0.0194 single-step IoU:
  erasing 20 % of TIR tokens is not a per-frame benefit.
* The free run's apparent +0.0046 over clean is the closed loop landing elsewhere, and its
  identical-history window is 0.0154 *below* clean.  So the TIR "improvement" of section 6.5 is
  closed-loop dynamics, not a property of the corrupted features.
* Forcing the **clean control path** (reliability and severity gates) onto the corrupted features
  is the worst arm (0.2778 mean, 0.7100 single-step): the damaged run's own reliability and gates
  are adapted to the damage, so the control path is active rather than inert.  The feature
  substitution moves nothing beyond noise, i.e. there is no evidence that TIR *content* is what
  hurts.
* **Correction of an earlier version of this section (and of the tool).**  The first measurement
  interleaved extra forwards that consume the global torch RNG *between* arms, which shifted the
  token-corruption masks of every later arm and every later sequence; it reported a free-run
  single-step IoU of 0.7178 against 0.6999 for the fixed crop, i.e. a larger effect than the
  matched-mask measurement supports.  `--seed` is now applied before **every** arm of every
  sequence, so all arms see the same masks, and the table above is the reseeded measurement.
* Caveats: 8 sequences (the first 8 of the stratified split), 40 frames, one checkpoint, one
  severity; per-sequence signs are mixed and only the aggregate is quoted.  The oracle-style
  "clean path" substitutions are diagnostics, not deployable methods.
* **Divergence is not only a corruption effect.**  The counter of section 6.15 is now part of a
  run's provenance (`tools/divergence_report.py` puts it next to every condition, and prints
  "unknown" rather than 0 % for runs recorded before the counter existed).  The first run measured
  with it is the MLP arm: even its `clean` run had **70 of 10 244 frames clamped (0.68 %)**, and
  over all five of its conditions 223 of 51 220 frames (0.44 %) were clamped.  A tracker that
  clips its own box on clean frames is a tracker whose loop is not numerically safe, which is why
  the bound is applied to every run rather than only to the divergent conditions.
* **One sequence can carry the divergence.**  In the MLP arm `rightgirltakingcup` is the worst
  sequence in four of the five conditions; a run-level rate of 0.44 % can therefore hide a single
  sequence that is effectively being tracked on a clamped crop, which is why the report prints the
  worst sequence per condition as well as the rate.
* Command (one table for every arm and family, with the pre-registered endpoint compared
  against the mask baseline):
  `python tools/family_report.py --arm mask=outputs/validation_v1/family_ --arm deviation=outputs/validation_v1/p4_deviation_family_ --families "burst:zeroing (seen),block:zeroing (held out),random:zeroing (held out),noise:additive noise (held out)" --primary noise --out outputs/family_report.json`.
* **Audit outcome**: the crop bound leaves the already-published token-condition numbers within
  +/-1 SR point (section 6.16 table), so those results stand; what it changes is that divergence is
  now measurable (3.44 % of frames on `tok_block_rgb_04`, 0.22 % on `tok_block_tir_04`).
* Command:
  `python tools/divergence_report.py --run mlp=outputs/validation_v1/ab_mlp_corr --run nodec=outputs/validation_v1/ab_nodec_corr --run full=outputs/validation_v1/ab_full --out outputs/divergence_report.json`.
* Command:
  `python tools/trajectory_replay.py --checkpoint outputs/ab_full/final.pth --sequence-list outputs/validation_split_v1/sequences.txt --sequences 8 --frames 40 --target tir --ratio 0.2 --intervene feature,control,both --seed 0 --out outputs/trajectory_replay_ab_full_tir_stage2.json`.

### 6.13 How much can 60 sequences actually detect? (`tools/power_plan.py`)

The reviewer's warning was that an MDE table built from assumed variances is decoration.  This
one is built from the measured per-sequence paired drops on the 60-sequence split
(`outputs/validation_v1/power_plan_pilot.json`), with the same multiplicity the final test
will carry (family of 8: condition comparisons + interactions), alpha = 0.05, power = 0.80:

| condition | metric | mean drop | sigma_drop | MDE at n = 60, m = 8 | verdict |
|---|---|---:|---:|---:|---|
| `tok_block_rgb_02` | SR | +3.44 pts | 10.63 | 4.91 | effect below the MDE |
| `tok_block_rgb_04` | SR | +9.03 pts | 17.00 | 7.85 | marginal, above the MDE |
| `rgb_occl_04` | SR | +10.99 pts | 18.24 | 8.42 | marginal, above the MDE |
| `tok_block_rgb_04` | PR | +11.72 pts | 23.06 | 10.65 | PR is noisier than SR |
| difference in differences `ab_nodec_corr - ab_full`, `tok_block_rgb_02` | SR | +0.24 pts | **16.46** | **7.60** | two orders of magnitude below the MDE |

Consequences for the experiment design (all from measured variance, not assumptions):

* **SR is the better primary endpoint than PR** at this sample size (smaller sigma relative to
  the effect); the pre-registered primary stays SR.
* Only the **strong** conditions -- RGB token block erasure at ratio 0.4 and image-level RGB
  occlusion at severity 0.4 -- produce effects that clear their own MDE at n = 60.  Ratio 0.2
  cannot be used as the primary test even though it is "significant" raw, which is consistent
  with it falling out after Holm (section 6.5).
* The **robustness claim itself** (difference in differences) has an MDE of ~7.6 SR points at
  n = 60 with m = 8.  A pre-registered single comparison (m = 1) lowers it to ~5.9 points, and
  reaching a 3-point effect would need roughly 385 sequences at this variance.  So the
  full-vs-no-decoder comparison on the current endpoint is only powered for *large* effects.
* Lower-variance endpoints are therefore worth measuring before committing to a long training
  campaign.  The evaluator now records **mean frame IoU per sequence** (`per_sequence[*].iou_mean`
  plus the aggregate `iou`) next to SR/PR/NPR, and `tools/power_plan.py --metric iou` runs the
  same paired, Holm-aware design calculation on it; a run that predates the field is rejected
  rather than averaged as zeros.  Mean IoU is not thresholded, so it should have the smaller
  per-sequence variance the robustness claim needs -- the number itself has to wait for the
  first evaluation runs that carry the field (the P2 arm grid), and the forward-decidable
  failure-time endpoint (`tools/horizon_probe.py`) is the other candidate to power.
* **Monte-Carlo power under the real Holm procedure** (`--simulate`), resampling the measured
  per-sequence differences -- no normality or variance assumption is added.  The simulated
  quantity is the full-vs-no-decoder difference in differences on `tok_block_rgb_02`, 60
  sequences, 800 repetitions per cell (`outputs/validation_v1/power_sim_pilot.json`):

| effect (SR points) | m = 1 | m = 4 | m = 8 |
|---|---:|---:|---:|
| 0 (FWER) | 0.069 | 0.020 | 0.015 |
| 2 | 0.129 | 0.031 | 0.009 |
| 3 | 0.274 | 0.095 | 0.050 |
| 5 | 0.661 | 0.446 | 0.324 |
| 8 | 0.985 | 0.929 | 0.866 |
| 12 | 1.000 | 0.999 | 0.999 |

  `tools/summarize_paired_conditions.py --metric iou_mean` runs the *same* paired, Holm-aware
  machinery, the screen and the composite on the mean-IoU endpoint, and every metric present in a
  run is written to the JSON; a sequence that lacks the field (a run recorded before the endpoint
  existed) shrinks `n` instead of crashing the comparison.
  The family-wise error rate is controlled (0.015-0.069; the m = 1 cell is the most liberal
  because the resampled distribution is heavy-tailed and the paired t-test is only asymptotically
  valid).  80 % power needs roughly 5-6 SR points at m = 1 and 6-7 at m = 8, so 1-3 point effects
  cannot be claimed from this design at all -- which is the quantitative form of the reviewer's
  "define the minimum credible evidence" before any long training campaign.
* Command:
  `python tools/power_plan.py --run ab_full=outputs/validation_v1/ab_full --run ab_nodec_corr=outputs/validation_v1/ab_nodec_corr --compare ab_full ab_nodec_corr --condition tok_block_rgb_02 --sequences 60 --comparisons 8 --out outputs/validation_v1/power_plan_pilot.json`.

### 6.14 The spatial-mixer control (`model.decoder_mode="spatial"`)

The reviewer's primary control question is whether an ordinary local spatial mixer explains the
gains, so the ablation arm now exists in the model rather than in prose:

* no parity matrix, no messages, no syndrome and no locator -- the branch never sees `parity` or
  `syndrome`, which a unit test pins down by changing both and asserting bit-identical output;
* each round mixes a token with its **3x3 neighbourhood** on the token grid (depthwise
  convolution, `groups=dim`) and then applies the same parameter-matched MLP as the `mlp` arm;
* the same `(1 - r) * gate` gating, the same output parameterisation (`post_norm` or
  `identity_residual`), the same budget, and the same supervision as the BP arm.

Parameter matching at the reference size (`dim=768`, 256 tokens, 2 rounds): BP 4.9277 M,
`mlp` 4.9288 M, `spatial` 4.9287 M (+0.02 %); the conv's 15.4 k parameters are traded for 5
hidden units per block (`mlp_hidden` 1600 -> 1595).  A non-square token grid is rejected instead
of silently falling back.  A unit test also checks that a perturbation of one token leaves every
token at Chebyshev distance >= 3 untouched -- i.e. the mixer is local, not a disguised global
one.

Training of this arm uses the same recipe and budget as the BP arms (6000 steps, no codec
warm-up, one seed) and lands in `outputs/p5_spatial/`; its step-size readout and the 60-sequence
tracking runs follow the same commands as section 6.10.

**Its result is section 6.25**: the spatial mixer matches the shipped BP arm and is
statistically indistinguishable from the best BP arm on robustness.  The scope decision below
records which conditions were run and why.

**Scope decision (cost).**  The four-arm grid and the spatial arm are evaluated on `clean`, the
calibrated degradation (`tok_block_rgb_04`) and one **non-zeroing** condition
(`tok_noise_rgb_04`, `tok_feat_noise` at ratio 0.4).  The random-erasure 0.2 condition is dropped:
its effect is indistinguishable from zero on the reference arm (section 6.5: +0.47 points,
interval spanning zero) and the block-versus-random contrast it was there for is already measured
on `ab_full`.

The non-zeroing condition was added *because* of section 6.7.1: every other token condition
erases, and on the erasing mechanism the shipped decoder is a ~2 % no-op (`recovery_gain` -0.02),
whereas on the non-zeroing mechanism it inflates the per-token error by 88 % (-0.88).  A
bounded-step, identity-residual arm is supposed to be unable to do that, so this is the condition
where the P2 parameterisation has the largest predicted effect -- testing the arms only on the
erasing mechanism would be testing them where the current decoder already does nothing.  Cost is
one more condition per arm, ~20 minutes each.

### 6.15 An unbounded crop window: one diverged frame cost 6.7 GB (`tools/` + trainer clamp)

While re-running the strongest token condition (`tok_block_erase`, target `both`, ratio 0.4) the
evaluation process sat at **GPU 0 %** while its RSS grew from 16.4 GB to 21.6 GB in ten seconds
and produced no metrics after 22 minutes.  The cause was not the model: `_crop_square` cropped a
square of the *predicted* side with no bound, so once the closed loop diverged and the box scale
exploded, `np.pad` allocated a window the size of the prediction.  Measured directly:

```
_crop_square(np.zeros((480, 640, 3), uint8), 320, 240, side=40000)  ->  shape (40000, 40000, 3)
                                                                        peak RSS + 6.7 GB
```

That is a pre-existing bug, not a P0-P5 artefact: any long evaluation whose tracker diverges can
hit it, and the failure mode is "GPU idle, CPU thrashing, no result" rather than a crash.

Fixed in two layers, both visible rather than silent:

* `_crop_square` clamps `side` to `MAX_CROP_SIDE_FACTOR = 8` times the frame's longer edge and
  clamps the centre onto the frame, so no caller can request an unbounded window (a legitimate
  search crop is 4x the frame, so the bound never touches a healthy run).  The same 40000-pixel
  request now costs 0.1 GB instead of 6.7 GB.
* `Trainer.infer_sequence` clamps every **predicted** box with `clamp_box` (finite, positive, at
  most `MAX_BOX_SCALE = 4` times the frame, centre on the frame) and **counts and logs** the
  frames it had to touch: `box_clamps` per sequence, `n_box_clamped_frames` in the summary.

The counter immediately showed how violent the condition is, and the finished run quantifies it:
`tok_block_both_04` (both modalities erased at 40 %, RGB-only mask for the report) needed the clamp
on **8 928 of 10 244 frames (87.15 %)**, its worst sequence on 197 of 200 frames (five sequences
above 190), and scored SR 3.70 % against a clean 33.95 % -- with both modalities erased at 40 % the closed loop does not
merely degrade, it dies.  Before the fix those frames would each have cropped a window the size of
a diverged prediction, which is why the run could not finish at all; the metric it now produces is
honest but describes a dead loop, so it is reported as a breakdown and kept out of the calibrated
condition set (6.5).  So metrics for the strongest conditions are partly a
statement about a diverged loop, which is exactly the distinction `tools/trajectory_replay.py`
(section 6.12) was built to make.

Consequences for reading earlier numbers: runs completed before this fix did **not** record clamp
counts, so their frames may include unbounded (diverged) crops.  The audit is done: the three
strongest token conditions of `ab_full` were re-run with the bound in place
(`outputs/validation_v1/ab_full_clamped/`, `tools/clamp_audit_report.py`,
`outputs/clamp_audit_report.txt`):

| condition | pre-fix SR | clamped SR | change | clamped frames | divergence | new mean IoU |
|---|---:|---:|---:|---:|---:|---:|
| `tok_block_rgb_04` | 24.92 | **25.45** | **+0.53** | 352 | 3.44 % | 0.2405 |
| `tok_block_tir_04` | 44.20 | **44.21** | **+0.01** | 23 | 0.22 % | 0.4338 |
| `tok_block_both_02` | 35.31 | **34.46** | **-0.85** | 97 | 0.95 % | 0.3338 |

**The published token-condition numbers stand.**  Every change is within +/-1 SR point, which is
far below this design's MDE (4-8 points, section 6.13), so the unclamped crop in the diverged
regime did not materially contaminate the results of section 6.5 -- the value of the fix was
(a) turning a no-result failure (`both@0.4`) into a measurement and (b) making divergence
*visible*: `tok_block_rgb_04` needs the clamp on **3.44 %** of its frames, a fact that could not be
reported before and that belongs next to its SR whenever it is quoted.

### 6.16 Decoder vs no decoder: the robustness claim is not measurable at this sample size

The pre-registered contrast of the whole project: does the error-correction branch make the
tracker **degrade less** under corruption than the identical network with the branch switched
off (`model.decoder_mode=off`)?  Both arms are the same checkpoint pipeline, same recipe, same
encoders, trained with the same budget; only the correction branch differs.  Difference in
differences over 60 paired sequences, `D = drop(nodec) - drop(full)`, so **positive means the
decoder arm degrades less** (`outputs/validation_v1/summary_full_vs_nodec.json`,
`tools/summarize_paired_conditions.py --interaction full nodec`):

| condition | drop, decoder arm | drop, no-decoder arm | D (pts) | 95% CI (pts) | p(t) |
|---|---:|---:|---:|---|---:|
| `tok_block_rgb_02` | +3.44 | +3.68 | **+0.24** | [-3.74, 4.55] | 0.911 |
| `tok_block_rgb_04` | +9.03 | +7.11 | **-1.92** | [-6.40, 2.77] | 0.415 |
| `tok_block_tir_02` | -4.74 | -7.15 | **-2.41** | [-6.99, 2.40] | 0.324 |
| `tok_block_tir_04` | -10.26 | -10.19 | **+0.07** | [-3.54, 3.68] | 0.972 |
| `tok_block_both_02` | -1.36 | -3.57 | **-2.21** | [-6.71, 2.20] | 0.340 |
| `rgb_occl_04` | +10.99 | +11.46 | **+0.48** | [-2.78, 3.85] | 0.780 |
| `rgb_lowlight_04` | +1.02 | +0.61 | **-0.41** | [-3.50, 2.64] | 0.796 |
| `tir_crossover_04` | +0.67 | +1.60 | **+0.93** | [-0.67, 2.74] | 0.291 |
| `tok_random_rgb_02` | +0.47 | -0.33 | **-0.80** | [-2.34, 0.50] | 0.282 |

**Correction (this table was first published with every D sign flipped).**  The estimator and
the JSON were always right (`D = drop_B - drop_A`), but the *printed* column was computed from a
leftover local `drop_a - drop_b`, so it disagreed in sign with the confidence interval printed on
the same line; the table above was transcribed from that column.  The print now reads the stored
estimate (single source of truth), the table is regenerated from
`outputs/validation_v1/summary_full_vs_nodec.json`, and every conclusion below is unaffected in
magnitude -- only the attribution of the (non-significant) advantage changes.

Holm-Bonferroni over the whole family of **27 tests** (18 condition comparisons + 9
interactions), as section 6.13 requires; nothing is significant, and the largest point estimate
is 2.41 SR points against confidence intervals of 3-7 points.

Two supporting facts:

* **Level check (no corruption).**  On `clean` the two arms differ by +0.38 SR points
  ([-3.25, 4.08], p = 0.84): the decoder neither helps nor hurts the in-distribution case, which
  is the manipulation check that the arms are otherwise comparable.
* **The TIR effect is not the decoder's.**  TIR token erasure *improves* SR by about 10 points,
  and it does so almost identically with the decoder off (-10.19) and on (-10.26), D = +0.07.
  Whatever produces that sign lives in the data/fusion path, not in the correction branch --
  consistent with section 6.12, where fixing the crops removed the effect.

**Reading.**  At 60 sequences x 200 frames with a 27-test family, the decoder's robustness
contribution is **not measurable**: the sign of D flips across conditions (the decoder arm
degrades less on `rgb_occl_04`, `tir_crossover_04`, `tok_block_rgb_02` and `tok_block_tir_04`, and
more on the other five) and every interval spans zero, which is exactly what section 6.13's
simulation predicts (80 % power needs 6-7 SR points; the observed |D| values are 0.07-2.41).  This is a statement about *this design*, not proof
that the branch is useless: it means the project cannot currently support the claim it wants to
make, and the honest report is "no evidence of improvement" rather than "improvement".

### 6.17 The minimum credible experiment (pre-registered design, derived from measured variance)

Sections 6.13 and 6.16 say the current 60-sequence design cannot support the robustness claim.
This is the design that can, computed from the **measured** per-sequence variance of the actual
contrast (`tools/power_plan.py`, alpha 0.05, power 0.80, Holm family of size m):

* **Primary condition**: `rgb_occl_04` -- image-level RGB occlusion, severity 0.4.  It is the
  largest and most consistent degradation (+10.99 SR points overall, +19.84 on the
  reference-tracking subset, Holm p = 0.0001) and, being an image-level condition, it does not
  share the token-corruption path that section 6.15 showed can break the closed loop.
* **Secondary condition**: `tok_block_rgb_04` (RGB token block erasure, ratio 0.4; +9.03 points,
  Holm p = 0.0009).
* **Primary metric**: SR, as pre-registered, with mean frame IoU as a co-primary endpoint (now
  recorded per sequence, section 6.13).  **Primary contrast**: the difference in differences
  between the decoder arm and the branch-off arm, on the primary condition, with **m = 1**;
  everything else is secondary and joins the Holm family.

| condition | sigma_D (SR pts) | MDE n=60, m=1 | MDE n=150, m=1 | MDE n=245, m=1 | MDE n=150, m=8 |
|---|---:|---:|---:|---:|---:|
| `rgb_occl_04` | 13.14 | 4.75 | **3.01** | 2.35 | 3.84 |
| `tok_block_rgb_04` | 18.12 | 6.55 | 4.14 | 3.24 | 5.29 |

Sequences needed at m = 1 for a given true difference in differences:

| condition | 1 pt | 2 pts | 3 pts | 5 pts | 8 pts |
|---|---:|---:|---:|---:|---:|
| `rgb_occl_04` | 1356 | 339 | **151** | 55 | 22 |
| `tok_block_rgb_04` | 2578 | 645 | 287 | 104 | 41 |

**Recommendation.**  `testingset` holds 245 sequences, so the actionable step is a 150-sequence
stratified split (2.5x the current one): at m = 1 that powers a 3-point effect on the primary
condition, and using all 245 would power 2.4 points.  Anything below ~2 points is out of reach at
any realistic size (a 1-point effect needs >1300 sequences at this variance), so the project
should state a 3-point minimum detectable effect up front and treat smaller differences as
"not measured" rather than as evidence of no effect.  The measured |D| values so far are 0.1-2.4
points (section 6.16), i.e. below even the enlarged design's threshold: the honest status is that
the correction branch's robustness contribution has **never been measured at a resolvable
effect size**, and the next training campaign should be decided on that basis rather than on
more point estimates from 60 sequences.

#### 6.17.1 The endpoint does not buy power (and what the tracker actually does)

The hope behind recording mean IoU per sequence (6.13) was that a non-thresholded endpoint would
have a smaller paired variance and therefore need fewer sequences.  Measured on the 60-sequence
split, that hope is **not supported**:

| quantity | mean | sd across sequences | median |
|---|---:|---:|---:|
| SR (AUC) | 0.3429 | 0.2797 | 0.2999 |
| mean frame IoU | 0.3313 | 0.2891 | 0.2849 |
| censored time-to-failure (fraction of horizon) | 0.3181 | 0.3296 | 0.1900 |

and the *paired* clean-level difference between two arms -- a pure noise floor, with no
corruption involved -- is essentially the same for both:

| endpoint | sd of the paired difference | MDE (n = 60, m = 1) | MDE (n = 150, m = 1) |
|---|---:|---:|---:|
| mean IoU | 0.0936 | 0.0338 | 0.0214 |
| SR | 0.0912 | 0.0330 | 0.0209 |

So switching endpoints does **not** cut the sample requirement; only enlarging the split or
shrinking the family does.  (`tools/summarize_paired_conditions.py --metric iou_mean` will run
the full paired/Holm/composite machinery on the endpoint once the runs that record it have
finished, so this conclusion can be re-checked rather than trusted.)

The failure-time endpoint is still worth reporting, because it says what the tracker *does* rather
than how it scores.  With the 20-consecutive-frames-below-IoU-0.5 rule on the 60-sequence split
(`outputs/horizon_probe_ab_full.json`):

* **78.3 %** of sequences fail within the 200-frame horizon;
* the **median failure frame is 16** (mean censored time-to-failure 63.6 frames of 200);
* once failed, only **12.8 %** ever recover.

**Wording correction (6.27.3).**  Three things this listing used to imply and should not.

1. "**Permanently**" was inferred from a 200-frame window.  47 of 60 sequences fail and 6 of those
   47 recover, so the observed quantity is **68.3 % (41/60) failing without recovering within the
   window**; nothing here speaks to behaviour past frame 200.
2. The **median failure frame of 16 is the median over the sequences that failed**, not a survival
   median over all sequences, and it is the *start* of the failing run.  The event is only knowable
   online `patience - 1 = 19` frames later, at a median confirmed frame of 35; the tool now stores
   both (`failure_frame`, `failure_confirmed_frame`) and prints the pair.
3. The claim "tracks for about 16 frames and then loses the target permanently on four sequences
   out of five" below is therefore too strong; the corrected reading is "on 41 of 60 sequences the
   target is lost within the window and not recovered inside it".

The clean SR of 0.34 is not "tracks at 0.34 quality for 200 frames": it is a loop that usually
loses the target early and then does not come back.  That is the most useful descriptive number in
this document for interpreting every other table, and it is why the arm contrasts are read as
*robustness* (how much worse corruption makes an already-fragile loop) rather than as an absolute
capability.

### 6.18 P4 pre-registration: mask vs deviation supervision, decided on a held-out mechanism

**Result: section 6.20** -- the deviation arm reached 0.8123 against the 0.7440 baseline, i.e. the
pre-registered decision rule was met.  The tracking follow-up is a separate, pre-registered
contrast (end of 6.20).

Section 6.7.1 turned the P4 question into a measurable one.  The detection head trained against
the injected mask reaches AUROC 1.0000 on every *zeroing* family (seen or not) and 0.7440 on the
held-out non-zeroing family `tok_feat_noise`, so the training signal teaches "this token was
zeroed" rather than "this token is damaged".  P4 replaces the target with the continuous feature
deviation from the frozen teacher, `q_i = e_i / (e_i + c_m)` with `e_i = mean_c |x_i - x*_i|`,
which is defined for any degradation that has a teacher.

**Pre-registered before the run** (the chain is `outputs/p4_queue.log`):

* **Arms**: `outputs/p4_deviation` (target `deviation`) against the existing `outputs/ab_full`
  (target `mask`).  Same recipe, budget, seed and corruption family for training
  (`tok_burst_erase`, ratio 0.2); only the detection target differs.
* **`c_m` is measured, not assumed**: `tools/calibrate_residual_budget.py` now reports the L1
  deviation quantiles and the suggested `c_m` (median of the damaged-token deviation), and the
  value used is written into `outputs/residual_budget_p4.json` and the resolved config.
* **Primary endpoint**: `reliability_auroc_rgb` on the held-out **non-zeroing** family
  `tok_feat_noise` (8 sequences x 40 frames, RGB only, ratio 0.2, severity 0.4), baseline
  0.7440.
* **Decision rule**: the deviation arm must *exceed* 0.7440.  A token-level bootstrap interval is
  deliberately **not** used as the criterion: tokens within a frame are correlated, so such an
  interval is far too narrow and would turn a 0.01 difference into "significant".  With 8
  sequences the design is only meant to separate a large effect (>= 0.05 AUROC) from nothing; a
  smaller difference is reported as "not resolved" rather than as an improvement.
* **Secondary readings**: the same AUROC on the zeroing family (must not collapse below the mask
  arm's 1.0000), the syndrome AUROC (which is at chance for the mask arm, 0.345-0.499), and the
  recovery diagnostics (`recovery_gain`) which must not regress.
* **What a positive result would and would not mean**: exceeding the baseline on a non-zeroing
  family shows the head is no longer a zero-token detector.  It does not show that the signal is
  useful for tracking -- that still requires the training-time ablation of section 6.17's
  primary contrast, at the sequence count that design demands.
* **Training-path smoke (done before queueing the chain).**  Six CPU steps with
  `loss.detect_target=deviation` and `detect_deviation_cm=0.25` complete, with finite losses and
  a saved checkpoint, so the soft-target path is exercised end to end rather than only in unit
  tests.  The logged reliability term is ~0.66 per modality, i.e. essentially the BCE of a
  maximum-entropy target: damaged tokens sit at `e ~ 0.21` (measured, section 6.7.1) and the
  pre-registered `c_m` is the median of that distribution, so `q ~ 0.46`.  That is deliberate --
  a median reference means "half the typical damage", and the endpoint is a *ranking* metric
  (AUROC), which does not care about the target's scale -- but it does mean the binary entropy
  floor dominates the logged number.  If the deviation-trained head fails to beat 0.7440, the
  first thing to try is a sharper `c_m` (a lower quantile), and that would be a new
  pre-registration rather than a post-hoc tweak.

### 6.19 First positive signal: the Tanner decoder beats a pointwise MLP of larger capacity

Section 6.16 says the decoder-vs-branch-off contrast is not resolvable at 60 sequences.  The
*architecture* contrast is different: the same checkpoint pipeline with the BP decoder against
the parameter-matched pointwise MLP arm (`outputs/ab_mlp_corr`, `decoder_mode=mlp`), both trained
under the same recipe, budget and corruption family.  Four conditions are shared -- the whole
token set the MLP arm was evaluated on -- and the sign is the same in all four
(`outputs/validation_v1/summary_three_arms.json`; `D = drop(mlp) - drop(full)`, positive = the BP
arm degrades less):

| condition | drop, BP arm | drop, MLP arm | D (pts) | 95% CI (pts) | p(t) | Holm p (36 tests) |
|---|---:|---:|---:|---|---:|---:|
| `tok_block_both_02` | -1.36 | +4.07 | **+5.43** | [1.45, 9.67] | 0.0137 | 0.355 |
| `tok_block_rgb_04` | +9.03 | +13.97 | **+4.94** | [1.06, 9.08] | 0.0194 | 0.447 |
| `tok_block_rgb_02` | +3.44 | +7.38 | **+3.94** | [0.36, 7.89] | 0.0474 | 0.900 |
| `tok_block_tir_02` | -4.74 | -1.67 | **+3.07** | [0.63, 5.68] | 0.0222 | 0.487 |

Averaging the four into a single pre-registered contrast (per-sequence mean drop over the
conditions, then the difference in differences -- `--composite`, so the number is reproducible by
a committed command rather than by an ad-hoc script):

| endpoint | drop, BP arm | drop, MLP arm | D | 95% CI | paired t p | seqs favouring BP |
|---|---:|---:|---:|---|---:|---:|
| absolute (SR points) | +1.59 | +5.94 | **+4.34** | [+1.75, +7.24] | 0.0034 | 0.72 |
| relative to clean (% of clean SR) | -12.65 | +3.13 | **+15.78** | [+6.74, +25.01] | 0.0013 | 0.73 |

with sigma_D = 11.03 points, i.e. MDE 3.99 at n = 60 (m = 1) and **2.52 at n = 150**, so the
observed effect is just above what the current split can detect and comfortably above what the
enlarged split can.

**How this may and may not be read.**

* It is a statement about the **decoder architecture**, not about the correction branch: the
  branch-on vs branch-off contrast (6.16) stays unresolved.  Read together, the two results say
  "which decoder you build matters; whether to build one at all is not measurable here".
* **The capacity confound runs against the result, not for it**: the MLP arm is the older
  checkpoint trained with the pre-fix auto width (`mlp_hidden=2048`, 6.307 M parameters against
  the BP decoder's 4.928 M), i.e. the BP arm wins with **28 % fewer** decoder parameters.
* **Read this together with 6.25, where the properly matched control lands.**  The `mlp` arm is a
  *pointwise* mixer; the parameter-matched *local spatial* mixer (P5) does not lose to the BP
  decoder at all (0.353 vs 0.339 clean; 0.252 vs 0.249 on `tok_block_rgb_04`), and against the best
  BP arm the robustness contrast is +0.10 points, p = 0.94.  So the honest summary of the two
  sections is: **the BP decoder beats a pointwise MLP, and does not beat an ordinary local spatial
  mixer** -- the gain in 6.19 is about having spatial structure at all, not specifically about the
  Tanner graph.
* **The level is not the explanation**: the MLP arm's clean SR is *higher* (34.57 vs 33.95), so a
  ceiling effect would work against the BP arm; the normalised endpoint, which removes the level,
  agrees (+15.8 % relative, p = 0.0013).
* **The family correction is the honest caveat**: inside the 36-test family of the whole
  three-arm table, Holm gives 0.355-0.900 and nothing survives.  That is not a contradiction --
  it is the argument of section 6.17 for pre-registering one contrast (this composite) instead of
  reporting a family of exploratory ones.
* Standing caveats unchanged: 60 stratified sequences, one seed, one checkpoint per arm, 200
  frames, and a corruption strength calibrated on the reference arm.
* Commands:
  `python tools/summarize_paired_conditions.py --run full=outputs/validation_v1/ab_full --run mlp=outputs/validation_v1/ab_mlp_corr --reference clean --composite full mlp --composite-conditions tok_block_rgb_02,tok_block_rgb_04,tok_block_tir_02,tok_block_both_02 --out outputs/validation_v1/composite_full_vs_mlp.json`
  (`--composite-relative` for the normalised row, and `--interaction full nodec` for the
  branch-off contrast of 6.16).

### 6.20 P4 result: a deviation target generalises where a mask target does not

The pre-registration of section 6.18 is now measured.  Two arms, identical recipe, budget, seed and
training corruption family (`tok_burst_erase`); only the detection target differs -- the injected
token mask (the shipped recipe) versus the continuous feature deviation
`q = e / (e + c_m)` with `c_m = 0.1626` **measured** as the median L1 deviation of damaged tokens
(`outputs/residual_budget_p4.json`, `tools/calibrate_residual_budget.py`).  Scored on 8 sequences x
40 frames, RGB only, ratio 0.2, severity 0.4 (`tools/family_report.py`,
`outputs/validation_v1/family_report_mask_vs_deviation.json`):

| arm | corruption family | mechanism | reliability AUROC | syndrome AUROC | locator p@5 | `recovery_gain` | error before -> after |
|---|---|---|---:|---:|---:|---:|---|
| mask (shipped) | `tok_burst_erase` | zeroing (seen) | 1.0000 | 0.373 | 0.838 | -0.0196 | 0.2115 -> 0.2157 |
| **deviation** | `tok_burst_erase` | zeroing (seen) | 1.0000 | **0.578** | 0.794 | **+0.1070** | 0.2157 -> **0.1926** |
| mask (shipped) | `tok_feat_noise` | additive noise (held out) | 0.7440 | 0.499 | 0.212 | -0.8754 | 0.1229 -> 0.2304 |
| **deviation** | `tok_feat_noise` | additive noise (held out) | **0.8123** | 0.501 | 0.204 | **-0.5574** | 0.1233 -> **0.1920** |

**Pre-registered decision** (6.18): the deviation arm must exceed 0.7440 on the held-out
non-zeroing family.  It reaches **0.8123, a delta of +0.0682** -- above the >= 0.05 effect size the
design was stated to be able to resolve, so this is a positive result by its own rule rather than
by a post-hoc threshold.

Three further readings, none of which were the primary endpoint:

* **It does not lose the easy case.**  On the zeroing family the deviation arm also measures
  1.0000, so the gain is not bought by giving up the zero-token cue.
* **The repair operator changed qualitatively on the training-like mechanism.**  `recovery_gain`
  goes from -0.0196 (a near no-op, error slightly *up*) to **+0.1070** (error genuinely down,
  0.2157 -> 0.1926).  This is the first configuration in the project that measurably repairs
  anything at all.
* **On the held-out mechanism it is still negative**, just less so (-0.8754 -> -0.5574): better,
  not solved.  The pair of numbers is the honest summary -- deviation supervision improves
  transfer and repair, and does not make the branch mechanism-independent.
* **The repair gain is not bought with collateral damage.**  The obvious alternative explanation
  for a positive `recovery_gain` is a more aggressive update that also damages healthy tokens; the
  diagnostics say the opposite (`damage_clean`, the error on untouched tokens):

| arm | family | `recovery_gain` | `damage_clean` |
|---|---|---:|---:|
| mask | `tok_burst_erase` | -0.0196 | 0.1251 |
| **deviation** | `tok_burst_erase` | **+0.1070** | **0.1055** |
| mask | `tok_feat_noise` | -0.8754 | 0.1297 |
| **deviation** | `tok_feat_noise` | **-0.5574** | **0.1051** |

  The deviation arm repairs more *and* perturbs healthy tokens ~16-19 % less, on both families, so
  the improvement is not an artefact of moving everything harder.

Not yet shown: that either improvement reaches the *tracker*.  That still requires the design of
section 6.17 (150 sequences, one pre-registered contrast), and the P2 arms test the other half of
the mechanism story (parameterisation and step size).  Caveats: 8 sequences, one seed, one
severity, token-level AUROC on a corpus the reference model never trained on but which is small
enough that a 0.07 difference is one design decision, not a law.

**Pre-registration of the tracking follow-up** (written before the run existed; it is the last
stage of the serial queue).  A `full` (mask) vs `p4` (deviation) contrast on the same 60
sequences, same conditions, same recipe, only the detection target differing:

* **Primary**: the difference in differences on **`tok_noise_rgb_04`** -- the held-out
  *non-zeroing* condition, i.e. exactly where the detection and repair advantages were measured.
  One condition, **m = 1**; positive means the deviation arm degrades less.
* **Secondary**: the same contrast on `tok_block_rgb_04` (the erasing condition, where 6.20 says
  the two arms differ less) and the `clean` level check.
* **Decision rule and its limit, stated up front**: this is powered to detect only a *large*
  transfer effect.  The paired difference-in-differences standard deviation for arm contrasts on
  this split measures 13-18 SR points (6.16, 6.17), so at n = 60 and m = 1 the MDE is roughly
  5-6.5 points.  A null result therefore means "no large transfer", not "no transfer"; the
  detection-side result of this section stands on its own endpoint, and a properly powered
  transfer test needs the 150-sequence split of 6.17.

### 6.21 Correction: the branch repairs its own configuration (and two training/eval gaps)

The "the boundary repair is a near no-op that sometimes *inflates* the error" statement of 6.7.1
and 6.20 was measured with `--corrupt-target rgb`, i.e. erasing **only the RGB** tokens.  The
training configuration is not that: it declares `corruption.token: tok_burst_erase`,
`ratio 0.2`, `severity 0.4` and **`target: both`** -- RGB *and* TIR tokens are erased -- and the
training log reports the hinge `gain 0.0000 act 0.00`, i.e. the repair constraint satisfied on
training batches.  That discrepancy was chased through four hypotheses, all rejected by
measurement, before the configuration itself was checked:

| hypothesis for "training says repaired, evaluation says worse" | test | verdict |
|---|---|---|
| the crops drift in the closed loop | `tools/trajectory_replay.py --recovery` with crops fixed to the clean trajectory | rejected (-0.0417 free vs -0.0386 fixed) |
| it is a domain effect (evaluation uses held-out sequences) | the same probe on `trainingset` sequences | rejected (-0.0385 free) |
| it is an fp16/AMP artefact | `tools/fp16_recovery_check.py`: the *training* loss on real batches in fp32, fp16 and eval mode | rejected (hinge satisfied in all three) |
| it is because evaluation crops are not centred on the target | the same probe replaying the ground-truth trajectory | rejected (-0.0378) |
| **the corruption target differs** | the same probe with `--target both`, i.e. the training token target (images clean, so a milder condition than training) | **accepted: recovery_gain +0.0553 free, +0.0499 fixed, positive on 83 % of sequences** |

So, with the configuration the model was actually trained on:

| corruption target | `recovery_gain`, free crops | `recovery_gain`, fixed crops | sequences positive |
|---|---:|---:|---:|
| `rgb` only (what 6.7.1/6.20 measured) | -0.0417 | -0.0386 | 0 % |
| **`both` modalities (the training token target)** | **+0.0553** | **+0.0499** | **83 %** |

Precision: the probe applies token erasure only, on clean images, whereas training also degrades
the search image (the dataset applies all four declared image corruptions).  The `both` row is
therefore the training *token* configuration under a strictly milder image condition -- a subset
of what training does, not an exact replica -- which is enough to explain the sign disagreement
with `rgb`-only, and is stated as such rather than as "the training setting".

**What this retracts and what stands.**  The claim "the branch does not repair" is wrong, and so is
"it inflates the error by 88 %" *as a statement about the training configuration* -- the latter was
measured on `tok_feat_noise`, a corruption family that is genuinely held out.  What stands, and is
now sharper: the branch repairs **exactly the configuration it was trained on** (both modalities
erased at ratio 0.2) and not single-modality erasure, not a non-zeroing family, and not
image-level degradation.  Mechanism specificity was the right conclusion; "it is a no-op" was an
artefact of evaluating a different configuration from the one that was trained.

**A claim I made here and then had to retract (kept visible on purpose).**  While checking the
first correction I grepped `Trainer.train()` for `apply_image_corruption`, found nothing, and wrote
that the image-level corruptions declared in the config are never applied in training -- so every
image-level condition would be out of distribution.  **That was wrong**, and a single measurement
showed it: the *dataset* applies them (`codetrack/data/datasets/lasher.py`, before cropping),
while the trainer does not need to.  Comparing one sampled item with the corruption switched off:

| tensor | identical with/without image corruption | mean abs difference |
|---|---|---:|
| `search_rgb` | no | 1.306 |
| `search_tir` | no | 0.291 |
| `template_rgb` | **yes** | 0.000 |
| `template_tir` | **yes** | 0.000 |

So the training recipe is: image-level corruption of the **search** frame (all four declared
types, severity 0.4, probability 1.0) plus token corruption of both modalities (ratio 0.2), with a
**pristine template** -- and the property is now pinned by
`tests/unit/test_dataset_corruption.py`.  The correct reading of section 6.5's image conditions is
the *opposite* of what I first wrote: each evaluation condition applies **one** corruption type,
while training applies **four at once**, so those conditions are in-distribution and *milder* than
training, not out of distribution.  The process lesson is worth more than the claim was: a
capability inferred from one file's grep is not evidence, and `apply_image_corruption` living in
the dataset rather than the trainer is exactly the kind of split that makes it easy to get wrong.

### 6.22 P2 tracking, arm A only (partial): the `post_norm` recipe is not reproducible in outcome

Arm A is the shipped recipe retrained with the current code (same config, seed, budget and
overrides as `ab_full`).  Its 60-sequence tracking runs are the first P2 arm to finish; B, C and D
are still running, so **nothing here is a statement about the parameterisation change** -- it is a
statement about the *baseline*, and it is uncomfortable.

| run / condition | SR | PR | NPR | mean IoU | clamped frames | divergence |
|---|---:|---:|---:|---:|---:|---:|
| `ab_full` (shipped) / `clean` | 0.340 | 0.410 | 0.380 | n/a | n/a | n/a |
| **`p2_A` / `clean`** | **0.290** | 0.360 | 0.330 | 0.2739 | 43 / 10 244 | 0.42 % |
| `ab_mlp_corr` / `clean` | 0.346 | 0.436 | 0.400 | 0.3338 | 70 / 10 244 | 0.68 % |
| `ab_full` / `tok_block_rgb_04` | 0.249 | 0.297 | 0.263 | n/a | n/a | n/a |
| **`p2_A` / `tok_block_rgb_04`** | **0.149** | 0.189 | 0.150 | 0.1347 | 0 / 10 244 | 0 % |
| `ab_mlp_corr` / `tok_block_rgb_04` | 0.209 | 0.263 | 0.229 | 0.1888 | 0 / 10 244 | 0 % |
| `p2_A` / `tok_noise_rgb_04` | 0.306 | 0.392 | 0.360 | 0.3007 | 0 / 10 244 | 0 % |

Paired over the 60 sequences (`tools/power_plan.py` conventions):

| contrast | mean | 95 % CI | p(t) |
|---|---:|---|---:|
| `clean`: A - shipped | **-5.18 pts** | [-8.57, -2.24] | 0.0024 |
| `clean`: A - MLP | **-5.80 pts** | [-9.09, -2.97] | 0.0005 |
| `tok_block_rgb_04` level: A - shipped | **-9.64 pts** | [-15.41, -4.27] | 0.0015 |

So the retrained `post_norm` arm is **significantly worse than the shipped checkpoint on clean
frames** and worse under corruption too.  This is the same arm whose update overshoots
`applied / oracle = 34.4` and pushes the damaged-token error up by 5.8 L1 points (6.10): the
recipe's output LayerNorm absorbs the update's scale, the scale is therefore nearly unconstrained,
and two runs of the same configuration land 20x apart in scale and 5 SR points apart in tracking.
That is a reproducibility problem in the shipped parameterisation, not a property of the
correction idea, and it is the strongest argument in this document for the bounded,
identity-residual output that arms C and D implement.

Two further readings that hold regardless of the pending arms:

* **Tokens are more forgiving than images here**: `tok_noise_rgb_04` leaves A at 0.306, *above* its
  clean 0.290.  Token-level noise perturbs the closed loop onto a different trajectory rather than
  damaging the appearance, which is consistent with 6.12 and with 6.20's mechanism story.
* **Divergence is not confined to heavy corruption**: A's own `clean` run needed the box clamp on
  43 frames (0.42 %) while both corruption conditions needed none, i.e. the clip is a property of
  the loop, not of the corruption level.

**Arm C (identity-residual, learned gate) is the best tracker on clean and on the non-zeroing
condition -- prediction 2 held, and it is the strongest tracking result so far.**  Full-set numbers
(all 60 sequences, same split, same evaluator):

| condition | shipped `ab_full` | MLP control | A `post_norm` | B `post_norm`, gate=1 | **C `identity_residual`** |
|---|---:|---:|---:|---:|---:|
| `clean` | 0.339 | 0.346 | 0.288 | 0.244 | **0.385** |
| `tok_noise_rgb_04` | (not run) | (not run) | 0.314 | 0.252 | **0.402** |
| `tok_block_rgb_04` | 0.249 | 0.206 | 0.153 | 0.108 | *run aborted, see below* |

Paired over the 60 sequences, C's clean level exceeds A's by **9.71 SR points** ([-15.14, -4.86] from
A's side, p = 0.0005) and its `tok_noise_rgb_04` level by **8.86 points** (p = 0.0005), and it is
**+4.6 points above the shipped checkpoint** on clean.  That is the direction the geometry of 6.10
predicted (`post_norm` arms raise the damaged-token error 32x; the identity-residual arm reduces
it), and it is the first tracking evidence that the parameterisation change is an improvement in
its own right rather than only a numerical-stability fix.

**Arm C's `tok_block_rgb_04` run failed and must not be quoted.**  Its log shows
`CUDA error: unspecified launch failure` and the evaluator skipped **51 of 60** sequences, so the
0.244 SR that the artifact reports is computed over 9 sequences.  The `n_skipped` field exists
precisely so this cannot pass unnoticed (my first reading of the table did not check it, which is
how it got one paragraph into this document).  The condition is being re-run with
`PARALLEL=1` after arm D's runs finish; a companion check is that the other two conditions of the
same arm completed all 60 sequences, so the failure is sequence-specific or transient rather than a
property of the arm.

**A vs B: removing the learned gate changes nothing measurable (prediction 3 held).**  Both arms
are the `post_norm` parameterisation with identical parameters; B was trained with the gate forced
to 1.  Paired over the 60 sequences (`outputs/validation_v1/summary_p2_A_vs_B.json`):

| condition | drop, A | drop, B | D (pts) | 95 % CI | p(t) |
|---|---:|---:|---:|---|---:|
| `tok_block_rgb_04` | +13.49 | +13.53 | +0.04 | [-3.71, 4.03] | 0.983 |
| `tok_noise_rgb_04` | -2.60 | -0.87 | +1.73 | [-0.18, 4.43] | 0.160 |
| **composite (two conditions)** | +5.44 | +6.33 | **+0.89** | [-1.60, +3.89] | **0.529** |

So the severity gate -- whose permutation effect on the update geometry is <= 0.001 in every arm
that does not force it (6.10) -- also has no measurable effect on *tracking* when it is removed
outright.  That is the third independent line of evidence that the second gate is decoration, and
it also shows what the two `post_norm` arms share: both drop about 13.5 SR points on
`tok_block_rgb_04`, against 9.0 for the shipped checkpoint from a higher clean baseline.

**Prediction, recorded before the remaining three arms land** (the geometry of 6.10 is known, the
tracking is not, so this is a falsifiable statement rather than a reading).  **Its outcome is
section 6.24**, where all four are scored: (1) not confirmed, (2) half confirmed, (3) half
confirmed, (4) the reading that applies.

1. **Primary (6.10 step 3)**: on the two token conditions, the identity-residual arms C and D
   should degrade **less** than the `post_norm` arms A and B, i.e. the A-vs-D `--composite` is
   positive.  The geometry predicts this in a way the design can actually resolve: the `post_norm`
   arms raise the damaged-token error from 0.1856 to **5.98 / 6.09** (a 32x increase), while the
   identity-residual arms *reduce* it to 0.171 / 0.170.  A contrast driven by a 32x difference in
   token-level damage is far above the ~5-point MDE of 6.17.
2. **Level check**: C and D should also track *better on clean* than A and B (whose update damages
   in-distribution tokens too -- that is the most plausible reading of A's -5.18 point clean
   deficit), and should be at or above the shipped checkpoint's 0.340.
3. **Gate**: A vs B and C vs D should be close to each other, because the gate is inert in every
   arm (permutation effect <= 0.001 for the three arms where it is not forced); the parameterisation,
   not the gate, is what changed.
4. **If 1 fails while the geometry is this different**, the conclusion is not "the parameterisation
   does not help tracking" but "the token-level repair does not propagate through the fusion and the
   tracking head", which would be a different and equally reportable finding.

What was missing before these predictions: arms B, C and D (now running with their overrides), the
single-factor contrasts of 6.10 step 3, and the `--composite` over the two token conditions.
`bash tools/p2_grid_report.sh` runs all of it once the four arm runs exist.

### 6.23 Process failure: an arm evaluated without its runtime overrides

While reading arm A's first tracking table I noticed that the P2 evaluation stage passed **no
runtime overrides**, so arms B, C and D were being evaluated with the default decoder settings:

* **B** was trained with `model.gate_always_one=true`.  With the gate forced to 1 the gating module
  receives no gradient, so its parameters stay at initialisation; evaluating B without the flag
  applies *untrained* gate weights.
* **C** and **D** were trained with `model.decoder_output=identity_residual`, which is what creates
  `branch_norm` and the zero-initialised `residual_out`.  Loading such a checkpoint into a
  `post_norm` model **silently drops those tensors** (they appear as unexpected keys) and evaluates
  a different network -- the load guard only checked `decoder_mode`, so nothing failed.

The affected runs were stopped after arm A had finished (A needs no overrides, so its results in
6.22 are valid), the invalid `outputs/validation_v1/p2_B/` directory was removed, and the remaining
work was moved to a corrected queue which passes each arm's overrides explicitly.  The arm-level
mistake was possible because the MLP arm in section 6.6 *did* carry `OVERRIDES`, which made it look
like the mechanism was in place.

Two fixes, so that the next occurrence fails loudly instead of producing a plausible number:

1. `load_checkpoint` now also refuses a mismatch in `decoder_output` and in `gate_always_one`, not
   just `decoder_mode`, naming the override to pass (pinned by a unit test that saves a checkpoint
   from an identity-residual and from a gate-forced model and asserts both mismatches raise).  The
   guard was then checked against the *actual* arm checkpoints, loading each into the default
   model:

| checkpoint | stored | loading into the default model |
|---|---|---|
| `p2_A` | `decoder_output=post_norm`, `gate_always_one=False` | accepted (it *is* the default) |
| `p2_B` | `gate_always_one=True` | **refused**: "trained with model.gate_always_one=True but the model is built with False" |
| `p2_C` | `decoder_output=identity_residual` | **refused**: "trained with model.decoder_output=identity_residual ..." |
| `p2_D` | `decoder_output=identity_residual`, `gate_always_one=True` | **refused** (same) |

   So the invalid evaluation would now stop at load time with the override to pass in the message,
   instead of completing and reporting a number for a network that was never trained.
2. The evaluation runner records its overrides in `summary["invocation"]` (already true) and every
   arm is now launched with an explicit `OVERRIDES` value, including the empty case, so "no
   overrides" is a decision rather than an omission.

**Second consequence, found by the guard itself.**  The *step-size probes* had been run the same
way, without overrides, so arms C and D were probed through a `post_norm` decoder (dropping
`branch_norm`/`residual_out`) and B without `gate_always_one`; the resulting geometry table is
corrected in 6.10, where the old numbers are kept in `outputs/step_size_probe_p2_*_invalid_arch.json`
and the conclusion reverses (C and D *do* repair; the `post_norm` arms are the ones that do not).

This is the second process bug of the review round found by reading a *number* rather than a log
(the first was the sign-flipped print of 6.16).  Both had the same shape: a plausible-looking
output that nothing in the pipeline contradicted -- and in this case the guard added for the first
occurrence immediately exposed the second.

### 6.24 P2 grid outcome: the geometry improved, the robustness contrast did not

All four arms are trained and (except one re-run) evaluated.  The pre-registered analysis of 6.10,
in order: update geometry first, then the single-factor contrasts, then the primary composite.

**Step 1 (geometry, 6.10) -- the parameterisation works.**  `applied / oracle` is 34.4 / 35.0 for
the two `post_norm` arms and **1.015 / 1.004** for the two identity-residual arms; `gain vs zero` is
-5.79 / -5.91 against **+0.0145 / +0.0151**, i.e. the identity-residual updates *reduce* the
damaged-token error (D: 0.1856 -> 0.1705) while the `post_norm` ones raise it sixfold.

**Step 3 (tracking) -- no contrast moves the robustness endpoint.**  All three pre-registered
single-factor composites, each one a single m = 1 contrast over the two token conditions
(`outputs/p2_report/composite_*.json`, run by `tools/p2_grid_report.sh`):

| contrast | question | D (pts) | 95 % CI | p(t) |
|---|---|---:|---|---:|
| A vs B | does removing the learned gate change the corruption response? | +0.89 | [-1.55, +3.87] | 0.529 |
| C vs D | the same question inside the identity-residual parameterisation | +0.84 | [-2.87, +4.35] | 0.651 |
| **A vs D** | **does the whole parameterisation change it?** | **-0.80** | **[-5.11, +3.40]** | **0.716** |

Per condition the primary contrast is `tok_block_rgb_04` -1.00 points ([-7.25, 5.15], p = 0.756) and
`tok_noise_rgb_04` -0.60 ([-4.47, 3.13], p = 0.761).  Every interval bounds any effect at roughly
4-5 SR points, and the null repeats on the mean-IoU endpoint (6.24 addendum below).

**What these contrasts are, and are not (6.27.1).**  All four arms A-D contain the correction
branch; every one of them is a BP arm.  A-vs-D changes the *output parameterisation* while
changing the gate setting at the same time, so it is a contrast between two correction
configurations, **not** "with versus without the error-correction module".  The module-level
comparison is 6.16 (`decoder_mode=off`) and it is likewise unresolved.  Reading -0.80 as "the
correction module buys nothing" is a category error, and the phrase was used that way in earlier
summaries of this section.  Arm A is additionally a `held`-semantics arm (see 6.10 correction and
6.27.1), so it can only be re-evaluated with `--override model.decoder_state_mode=held`.

**The "34.4 / 35.0" figures quoted in step 1 are `L1(applied)/L1(oracle)`, an error ratio**, not a
step-size overshoot factor; the step ratio `a_applied/a*` is a separate number (6.27.3).  The
mechanism claim -- that the `post_norm` update *raises* the damaged-token error while the
identity-residual one *lowers* it (0.1856 -> 0.1705, an 8.1 % reduction) -- is unaffected, because
that is a statement about which direction the error moves, not about how the two ratios are named.

**Prediction 1 of 6.22 is therefore not confirmed.**  Despite a 32x difference in token-level
damage, the two arms degrade by the same amount under corruption.  Prediction 4 of 6.22 applies
verbatim: the conclusion is *not* "the parameterisation is worthless" but "the token-level repair
does not propagate into the robustness of the tracker" -- and the interval excludes effects larger
than ~5 SR points, so this is a bound rather than an absence of any effect.

**Where the arms do differ: clean tracking, and there the gate matters.**
`outputs/validation_v1/p2_*` clean runs, all 60 sequences:

| arm | parameterisation | gate in training | clean SR | PR | mean IoU | clamped frames |
|---|---|---|---:|---:|---:|---:|
| A | `post_norm` | learned | 0.288 | 0.357 | 0.2739 | 43 (0.42 %) |
| B | `post_norm` | forced to 1 | 0.244 | 0.316 | 0.2280 | 0 |
| **C** | **`identity_residual` + clip** | **learned** | **0.385** | **0.467** | **0.3742** | 189 (1.84 %) |
| D | `identity_residual` + clip | forced to 1 | 0.277 | 0.369 | 0.2630 | 0 |

So the *combination* is what wins: identity-residual output with the learned gate left in place is
the best tracker of the four by a wide margin (**+4.6 SR points over the shipped checkpoint**,
+9.7 over arm A), while the same parameterisation with the gate forced to 1 (D) is no better than
the `post_norm` arm A.

**The gate changes the level, not the drop -- and that is the cleanest statement the grid supports
about it.**  Level differences (paired, `outputs/p2_report/level_check.txt`), i.e. how much better
the learned-gate arm tracks at all, in every condition:

| pair | `clean` | `tok_block_rgb_04` | `tok_noise_rgb_04` |
|---|---:|---:|---:|
| A - B (learned minus forced, `post_norm`) | **+4.40** [0.37, 8.53] p = 0.040 | +4.44 [1.69, 7.78] p = 0.007 | +6.13 [2.96, 9.82] p = 0.001 |
| C - D (learned minus forced, identity-residual) | **+10.76** [5.81, 16.30] p = 0.0002 | +13.89 [9.57, 18.62] p < 0.0001 | +9.31 [4.35, 14.53] p = 0.0007 |

Compare with the composites above: removing the gate *at test time* changes the update geometry by
<= 0.001 and the corruption response by ~0.9 points, yet *training* with it forced to 1 leaves the
model 4.4 points lower in the `post_norm` parameterisation and **10.8 points lower** in the
identity-residual one.  So the learned gate is not a damage detector at all (that was already
established); it acts as a training-time conditioner, and the parameterisation that makes the gate
most useful is the identity-residual one.  "The gate is decoration" was too strong and is corrected
here: it is decoration *for the robustness contrast*, and worth 10 points of clean performance in
the arm where the rest of the design is right.

**Status of the four predictions of 6.22**: (1) primary composite positive -- **not confirmed**
(-0.80 points, CI [-5.10, +3.37]); (2) identity-residual arms above the `post_norm` arms on clean --
**half confirmed** (C yes, +9.71 over A, p = 0.0005; D no); (3) A ~ B and C ~ D -- **half
confirmed** (null on the corruption contrasts, but B is 4.4 points below A and D is 10.8 below C on
clean); (4) if (1) fails, read it as a propagation failure rather than a dead end -- **that is the
reading recorded above**.

**The null is not an artefact of the SR threshold.**  Repeating the primary composite on the mean
frame-IoU endpoint (`--metric iou_mean`, recipe step 6) gives D = **-0.87 IoU points**
([-5.31, +3.46], p = 0.70) -- the same null with the same direction, so the endpoint change of 6.17.1
does not manufacture a difference here either.

**A third axis, and it goes the other way: the identity-residual arms diverge more.**
`tools/divergence_report.py` over the four arms (frames clamped at the crop bound, and the worst
sequence in each condition):

| arm | `clean` | `tok_block_rgb_04` | `tok_noise_rgb_04` | worst sequences |
|---|---:|---:|---:|---|
| A `post_norm` | 43 (0.42 %) | 0 | 0 | `rightgirltakingcup` |
| B `post_norm`, gate=1 | 0 | 0 | 0 | -- |
| C `identity_residual` | **189 (1.84 %)** | 3.87 % (aborted run) | **119 (1.16 %)** | `shinycarcoming2`, `carcomingfromlight` |
| D `identity_residual`, gate=1 | 0 | **404 (3.94 %)** | 70 (0.68 %) | `boyruninsnow`, `shinycarcoming2` |

The two arms whose updates actually repair tokens are also the two whose closed loop needs the box
clamp most (1.54-1.67 % of frames against 0.00-0.14 % for the `post_norm` arms), and one sequence
(`shinycarcoming2`) is the worst case in three different arm/condition pairs.  Whether a stronger
repair amplifies the loop, or whether both are symptoms of a less constrained decoder, is not
settled by these runs; what is settled is that the parameterisation change is not free, and that
"mean IoU" or "SR" for those arms has to be read next to their clamp rate.  It also means the
four-arm ranking is not one-dimensional: B is the most numerically stable arm and the worst tracker,
C is the best tracker and one of the two least stable.

Open items: arm C's `tok_block_rgb_04` is being re-run (its first run hit a CUDA launch failure and
scored only 9 of 60 sequences, 6.22), which is what the C-vs-D and A-vs-C composites need.  Caveats
unchanged: 60 sequences, one seed per arm, one severity, and a clean-level difference that is *not*
controlled between arms (which is why the primary endpoint is the paired corruption contrast rather
than the level).

### 6.25 The spatial-mixer control: ordinary local context accounts for the results

The P5 question was whether an ordinary local spatial mixer -- no parity matrix, no messages, no
syndrome, no locator, parameter-matched to +0.02 % (6.14) -- can explain what the Tanner-graph
branch is credited with.  It was trained with the same recipe and budget and evaluated on the same
three conditions (`outputs/validation_v1/p5_spatial/`, `tools/p2_grid_report.sh` machinery):

| run | clean SR | `tok_block_rgb_04` | `tok_noise_rgb_04` | mean IoU (clean) | divergence (clean) |
|---|---:|---:|---:|---:|---:|
| shipped `ab_full` (BP, `post_norm`) | 0.339 | 0.249 | (not run) | n/a | n/a |
| MLP control (`decoder_mode=mlp`) | 0.346 | 0.206 | (not run) | 0.3338 | 0.68 % |
| **arm C (BP, `identity_residual`, learned gate)** | **0.385** | **0.291** | **0.402** | **0.3742** | 1.84 % |
| **spatial mixer (`decoder_mode=spatial`)** | **0.353** | **0.252** | **0.376** | **0.3412** | 0.78 % |

Paired over the 60 sequences, C minus spatial (`outputs/validation_v1/summary_C_vs_spatial.json`):

| contrast | difference | 95 % CI | p(t) |
|---|---:|---|---:|
| `clean` level | +3.15 pts | [-1.19, 7.82] | 0.178 |
| `tok_block_rgb_04` level | +3.93 pts | [0.25, 7.76] | 0.044 |
| `tok_noise_rgb_04` level | +2.58 pts | [-1.67, 7.36] | 0.267 |
| **robustness composite (m = 1)** | **+0.10 pts** | **[-2.50, +2.55]** | **0.937** |

**The answer is that local spatial context accounts for almost all of it.**  On the robustness
contrast -- the endpoint the whole project is built around -- the parameter-matched spatial mixer
and the designed BP arm are **indistinguishable** (+0.10 points, CI spanning zero), and on the
level the BP arm is ahead by 2.6-3.9 points, of which only one condition reaches nominal
significance.  Two readings follow:

* Everything the shipped checkpoint achieves on tracking is reachable by a mixer whose *direct
  message update* is a 3x3 depthwise convolution.  What the spatial arm replaces is the
  variable-to-check/check-to-variable message passing, not the whole system.
* The one configuration that is ahead of the spatial control is arm C -- and its advantage comes
  from the **output parameterisation plus the learned gate**, not from the graph: the same graph with
  `post_norm` (A) is 6.5 clean points *below* the spatial mixer, and the same graph with the gate
  forced off (D) is 7.6 points below it.

**Scope correction (6.27.1).**  An earlier version of this section concluded that "the Tanner graph
is therefore **not necessary**".  That overstates what this arm can show, for two independent
reasons.

1. The spatial arm is **not** a Tanner-free system.  `codetrack/models/codetrack.py` runs
   `H -> parity -> Tanner -> syndrome -> locator -> gate` unconditionally, whatever
   `decoder_mode` is, and hands the resulting gate to the decoder.  The spatial mode swaps the
   *message update* only; syndrome, locator, incidence and gating are all still there.
2. The arm was built with the default `post_norm` output, which its own step probe shows overshoots
   by ~2.8x.  So "spatial context is sufficient" is measured on a spatial arm that is handicapped by
   the parameterisation the same document shows is broken.

The defensible statement is: **under the current control path and recipe, ordinary spatial mixing
substitutes for the direct BP message update with no measurable loss.**  Negating the whole Tanner
system needs a second arm in which the gate is produced by something that does not consume
syndrome or locator at all; the parameterisation-matched spatial arm (spatial + identity_residual +
learned gate) is the first step and is pre-registered in 6.27.4.

Also note that p = 0.937 is not a demonstration of equivalence.  The interval is [-2.50, +2.55]
points, so the supported claim is "no difference larger than about 2.5 points", and only if a
+/-3-point equivalence bound is accepted *in advance*.  With the measured sigma_D the design cannot
resolve anything smaller (6.17).

**What this control does not settle**: whether *adding* the graph to a model that already has the
identity-residual output and a learned gate buys anything over the spatial mixer with the same
output parameterisation.  That arm -- spatial mixer + `identity_residual` + gate -- has not been
trained, and it is the honest next experiment: without it, "the error-correction view contributes"
cannot be separated from "the output parameterisation contributes", because the spatial arm was
built with the default `post_norm` output (its own geometry is 2.80x overshoot, 6.10).

### 6.26 P4's tracking follow-up: better detection and repair, worse robustness

Section 6.20 pre-registered this contrast: the mask-trained arm (`ab_full`) against the
deviation-trained arm (`p4_deviation_track`), same recipe, same budget, only the detection target
different, with the difference in differences on the held-out **non-zeroing** condition
`tok_noise_rgb_04` as the primary endpoint and `tok_block_rgb_04` as secondary.

Levels (60 sequences):

| run | `clean` | `tok_block_rgb_04` | `tok_noise_rgb_04` |
|---|---:|---:|---:|
| `ab_full` (mask target) | 0.339 | 0.249 | 0.363 |
| `p4_deviation` (deviation target) | 0.329 | **0.177** | 0.342 |

Difference in differences, `D = drop(p4) - drop(full)`, positive = the mask arm degrades less:

| condition | drop, mask arm | drop, deviation arm | D (pts) | 95 % CI | p(t) |
|---|---:|---:|---:|---|---:|
| `tok_noise_rgb_04` (**pre-registered primary**) | -2.35 | -1.23 | **+1.11** | [-0.57, 2.97] | 0.227 |
| `tok_block_rgb_04` (secondary) | +9.03 | **+15.22** | **+6.19** | [1.02, 11.77] | **0.028** |
| composite over both (m = 1, **exploratory**) | +3.34 | +6.99 | **+3.65** | [1.01, 6.48] | **0.013** |

**Correction (6.27.3): the composite row is not pre-registered.**  6.20 fixed the primary endpoint
(`tok_noise_rgb_04`) and one secondary (`tok_block_rgb_04`); averaging the two into a single
contrast was decided when the numbers were in, and `tools/summarize_paired_conditions.py` now marks
every `--composite` result `in_holm_family: false` and says so on stdout.  The composite is an
honest summary of this sample and it is the row that excludes zero, but it may not be quoted as a
pre-registered test, and the pre-registered verdict on this contrast is the *primary* row: no
difference.  The two rows are therefore reported together and labelled, not merged.

**Pre-registered decision**: the primary endpoint does **not** confirm a difference (+1.11 points,
interval spanning zero), and the compound reading over both conditions puts the deviation arm **3.65 SR
points worse** with an interval that excludes zero (p = 0.013).  So the honest verdict is not
"the deviation target hurts tracking" on the pre-registered condition alone, but "the arm with the
better detector and repairer is not better and is measurably worse in aggregate".

**The arm whose detector and repair are measurably better (6.20: held-out non-zeroing AUROC 0.8123
vs 0.7440, `recovery_gain` +0.107 vs -0.020 on the training mechanism, less collateral damage on
healthy tokens) tracks *worse* under corruption** -- it loses 15.2 points where the mask-trained
arm loses 9.0, a difference of 6.2 points with an interval that excludes zero.  Clean levels are
comparable (0.329 vs 0.339), so this is not a level artefact.

That is the same pattern as 6.24, from an independent manipulation: **no tracking benefit followed
from improving the token-level quality of the repair, and on this sample the arm with the better
repair tracked worse in aggregate.**  In 6.24 the geometry was fixed and the robustness contrast
did not move; here the detection and repair were improved by a different loss and the robustness
contrast moved against the arm with the better repair.  Two independent changes to the correction
mechanism, neither producing the expected tracking gain, is a stronger statement than either alone.

**How far that statement goes, and where it stops (6.27).**  An earlier version of this paragraph
concluded "whatever limits this tracker, it is not the quality of the per-token correction".  That
is stronger than the measurements support, for four reasons that are each individually enough:

* **The A-vs-D contrast of 6.24 is not a module switch.**  Both arms contain the correction branch,
  and they differ in the output parameterisation *and* the gate setting at once, so it cannot
  attribute anything to the module (6.27.1).
* **`L1(applied)/L1(oracle)` is an error ratio, not a step calibration.**  Reducing it from 34 to
  1.0 while the error falls 8.1 % is a real mechanism improvement, but it does not show the step was
  matched to `a*` (6.27.3).
* **The two manipulations do not act on one mediator.**  `identity_residual` changes scale,
  initialisation and output distribution; the deviation target changes the reliability calibration,
  and the control variable is `(1 - r)`, so a better AUROC ranking can coexist with a worse
  effective update magnitude.
* **The teacher is token-clean, not image-clean** -- the dataset applies image-level corruption
  before the backbone -- so all of these numbers are about repairing *token* damage in an
  already-degraded representation.  A repair that improves background tokens, or tokens in the
  teacher's coordinate frame, need not improve foreground localisation.
* **The auxiliary objective may be actively harming tracking rather than merely failing to help.**
  With `lambda_correct = 2.0` and `lambda_preserve = 0.5`, the gradient the shared weights receive is
  `g_track + Σ λᵢ gᵢ`; the unweighted `cos(L_track, L_correct)` that was the only thing measured
  cannot separate those two cases.  The weighted diagnostic now can (6.27.3), and on a 25-step
  smoke run the auxiliary gradient is already 2.07x the tracking gradient and opposed.

So the supported claim is: **no transfer of token-level repair quality to tracking robustness has
been observed**, and the mechanism that limits this tracker has not been identified.  Distinguishing
"repair is ineffective" from "repair is harmful" is one of the pre-registered items of 6.27.4.

**A gap in my own pre-registration, stated plainly -- and closed.**  The primary endpoint was
`tok_noise_rgb_04`, but `ab_full` had no run for it: the condition was introduced *after* `ab_full`'s
matrix was complete and had only ever been run for the P2 arms and the P4 arm, so the primary
contrast did not exist when the follow-up finished and only the secondary condition could be read at
first.  The missing reference run was executed (6 minutes, `outputs/validation_v1/ab_full_noise/`,
installed as `ab_full/tok_noise_rgb_04/`) and the primary row above is its result.  A
pre-registration that names a condition nobody runs is not a pre-registration; the check that would
have caught it -- does every arm have every condition the protocol names? -- is the same class of
gap as 6.23, and it is worth a guard in the runner (every condition in the protocol listed for every
arm, with a missing cell reported as a missing cell rather than silently omitted from a comparison).

### 6.27 Round-2 truth-fix, provenance, and pre-registration

A third review (of `073c223`) found defects that change how earlier numbers must be read, and the
fixes cost no training.  They come first because a comparison whose two arms were evaluated by
different code is not a comparison, and enlarging the sample cannot repair a mislabelled
estimator.  Everything in this section is produced by a script under `tools/`.

#### 6.27.1 The default decoder path was **not** bit-identical after `073c223`

`073c223` replaced `v` with `branch_input` inside the decoder loop but refreshed `branch_input`
only when `output_mode == "identity_residual"`.  For every other mode that changed the *function*:
from round 2 on, the message and update functions read the decoder's **input** instead of the state
round 1 produced, so the iterations stopped composing.  Nothing moved in the state dict, so
`load_checkpoint` could not notice, and the commit message -- and this document -- described the
default `post_norm` path as unchanged.  It was not.

`tools/decoder_state_equivalence.py` measures it.  It builds one decoder, copies the weights into
three instances and runs identical inputs through (a) `recurrent` -- the shipped default, (b)
`held` -- what `073c223` silently did, and (c) an **independently written re-implementation** of
the pre-`073c223` loop, so the check is not circular.  `outputs/decoder_state_equivalence.json`:

| mode | output | rounds | `max\|recurrent − held\|` | `max\|recurrent − parent\|` |
|---|---|---:|---:|---:|
| bp | post_norm | 1 | 0 | **0** |
| bp | post_norm | 2 | 0.082 | **0** |
| bp | post_norm | 3 | 0.240 | **0** |
| mlp | post_norm | 2 | 0.093 | **0** |
| spatial | post_norm | 2 | 0.138 | **0** |

`recurrent` reproduces the pre-`073c223` loop bit for bit on every `post_norm` cell, and the two
modes are bit-identical at one round, which localises the break to multi-round state.  On outputs
with unit scale a difference of 0.08-0.24 is a real change of function, not numerical noise.

**Consequences, all of which are now explicit.**
* `model.decoder_state_mode` is a config key with a `load_checkpoint` guard, exactly like
  `decoder_output` and `gate_always_one`.  The default is `recurrent`.
* The P2 arms **A and B** (and the shipped `post_norm` lineage) were trained under one semantics
  and, wherever they were evaluated after `073c223`, evaluated under the other.  Arms C and D
  (`identity_residual`) always refreshed, so they are self-consistent.  Rebuilding A or B needs
  `--override model.decoder_state_mode=held`.
* **The provenance paragraph of 6.10 is superseded.**  It attributed arm A's mismatch with
  `ab_full` to "the corruption/sampling RNG stream having shifted".  That explanation was never
  checked against the one difference that is guaranteed to change the loss: the decoder loop's
  state semantics.  The correct status is *not attributed*: the mismatch has at least two candidate
  causes, and identifying which needs the equivalence run plus a matched re-evaluation.
* The claim "the default path is bit-identical", wherever it appears, is **retracted**.

#### 6.27.2 Nothing in the repository could be attributed to a commit

`tools/run_provenance_audit.py` walks `outputs/` for training provenance and evaluation manifests
and reports, per run, the training commit, the evaluation commit, the dirty flag and the function
switches.  First run (`outputs/run_provenance_audit.json`): **74 runs, 74 not comparable** -- every
existing number predates the schema, so none of them can be tied to a commit.  This is not a
surprise given the schema is new, but it is the reason the round-2 numbers have to be produced
again rather than extended.

The schema now travels with every run: `run_provenance.json` for training (commit, dirty flag,
`decoder_state_mode`, `decoder_output`, `gate_always_one`, active parameters, peak memory) and
`eval_provenance` plus the checkpoint's own `train_provenance` inside every evaluation
`run_manifest.json`.  Two consequences for the round-2 experiments: **the provenance of the two
arms of any comparison must match**, and **a dirty tree invalidates attribution** even when the
commit matches.

#### 6.27.3 What the earlier metrics actually measured

Four names were wrong or too coarse.  Each now has a separate measurement, because in every case
the old single number was read as the narrower one.

| was called | actually was | now also reported |
|---|---|---|
| "applied/oracle = 34.4 (step overshoot)" | `L1(applied)/L1(oracle)`, an **error** ratio (`tools/compare_step_probes.py:56`) | `a_applied/a*`, the multiplier ratio along the ideal direction (`tools/step_size_probe.py`) |
| "N frames clamped (diverged)" | four events in one counter: scale, centre, non-positive, non-finite (`codetrack/engine/trainer.py:clamp_box`) | `n_box_clamps_by_reason` per run, `box_clamps_by_reason` per sequence, and a `scale rate` column in `tools/divergence_report.py` |
| "the crops are identical" | `allclose(frame 0 predictions) or len(gt) > 0`, i.e. always true | crops recorded per frame by the loop; `crop_identical` (frame 0) and `crop_identical_frames` (how far the loops stayed together) |
| "forward-decidable failure at frame *t*" | the **start** of the failing run, knowable only at `t + patience − 1` | `failure_confirmed_frame`, plus an RMST endpoint (`tools/horizon_probe.py`) |

Two of these had a second defect behind them.

* **The fixed-crop reference was misaligned by one frame.**  The free loop crops frame *f* around
  the box produced for *f − 1* (and around the frame-0 annotation for *f = 0*), never around its own
  output for *f*.  Passing `reference[f]` gave the replay a crop the free run never saw and handed
  it the reference's answer one frame early.  `crop_box_for_frame` now lags the schedule, and frame
  0 stays annotation-driven in every arm.  **Every fixed-crop result is therefore provisional**,
  including the TIR reading of 6.12; the re-run is part of the round-2 plan, not optional.
* **The `power_plan.py` FWER row was the wrong probability.**  `simulate_power` checked
  `0 in holm_rejected(...)`, i.e. P(this hypothesis is rejected), not P(any rejection in the
  family) -- so the "FWER" row of 6.13 was roughly alpha by construction.  It also drew an
  independent sample per comparison, destroying the positive correlation between contrasts measured
  on the same sequences.  `family_wise_rejection_rate` now counts *any* rejection, and one resample
  is shared across the family (an `(m × n)` effect matrix can be passed for the realistic joint
  structure).  The corrected numbers are conservative where the old ones were optimistic, so
  **6.13's power figures are superseded and must be regenerated.**
* `--composite` in `tools/summarize_paired_conditions.py` writes `in_holm_family: false` and says so
  on stdout.  The composite is defined by the conditions named on the command line, so it is not in
  the Holm family that the same script corrects; 6.26's "+3.65 points, p = 0.013" came from this
  path and was read as if it carried the primary endpoint's pre-registration.  See 6.27.4.
* `tools/gradient_conflict_report.py` reported only the **unweighted** `cos(L_track, L_correct)` on
  three modules, which cannot separate "repair is ineffective" from "repair is harmful": the
  optimiser sees `g_track + Σ λᵢ gᵢ`.  The trainer now logs the weighted reading too, and the
  report aggregates it (mean, minimum, ratio `‖g_aux‖/‖g_track‖`, by thirds of the run, and the
  fraction of steps that are *opposed and loud*).  On the 25-step smoke run the unweighted pair
  looks benign (mean cos −0.03, gradient ratio 0.87) while the weighted auxiliary gradient is
  **2.07× the tracking gradient** and opposed at every logged step.  That is a smoke run, not a
  result, but it is the measurement 6.27.4 needs.

#### 6.27.4 Round-2 pre-registration (written before the runs)

Fixed now, so the round-2 results cannot be read selectively afterwards.

* **Primary contrast**: the decoder arm against the identical network with the correction branch
  switched off (`model.decoder_mode=off`) -- the project's own claim, and **not** the A-vs-D
  parameterisation contrast of 6.24.  Primary condition `rgb_occl_04`, secondary
  `tok_block_rgb_04` (as in 6.17).  **m = 1**; every other condition comparison and interaction
  joins the Holm family.
* **Minimum useful effect: 3 SR points.**  Anything smaller is reported as "not measured", not as
  evidence of no effect, and sample sizes are computed from the *measured* σ_D of the contrast
  being tested (106-287 sequences over the observed 11-18 range), never from the smallest σ.
* **Secondary endpoints, pre-declared as mechanistic**: RMST up to τ and the confirmed failure
  fraction (`tools/horizon_probe.py`), and mean frame IoU.  None of them replaces the primary SR
  contrast; switching the *primary* endpoint after seeing results is not permitted.
* **Exploratory by construction** (must be labelled as such wherever reported): the composite over
  two conditions, any contrast on a condition introduced after the fact, and any comparison whose
  two arms differ in a function switch (`decoder_state_mode`, `decoder_output`, `gate_always_one`)
  without the matching override.
* **Fairness requirements for any "the module helps" statement**: matched initialisation for shared
  modules, matched optimizer steps *and* effective batch *and* precision *and* sample exposure, the
  measured active-parameter count, and FLOPs/latency.
* **Order of work**, decided here so a null result cannot be chased with a new endpoint: fix the
  measurements (this section), then the zero-training diagnostics, then the two training contrasts,
  then replication, then the enlarged split.
* **Standing limitation**: one seed per arm and one severity.  A single-seed p-value contains no
  information about *training* variance; seed variance is reported separately from sequence
  variance wherever a mechanism claim is made.

| pre-registered contrast | arms | endpoint | state |
|---|---|---|---|
| decoder vs branch-off | `ab_full` vs `ab_nodec_corr` (re-evaluated under matching semantics) | DiD on `rgb_occl_04`, m = 1 | not yet run under matching provenance |
| LoRA vs frozen, at fixed capacity budget | F0/F1/A0/A1 | the interaction `(A1−A0) − (F1−F0)` | not started |
| spatial+identity_residual+gate vs arm C | new arm vs `p2_C` | robustness composite | not started |

### 6.28 Where the failure actually is: the crop trajectory, not the tokens

Sections 6.24 and 6.26 improved the repair mechanism twice and the robustness contrast did not move
(once against the improved arm).  That raised the question 6.27.4 pre-registers: is per-token repair
quality even a mediator for tracking?  Neither earlier experiment ever handed the tracker a
*repaired* token set -- both changed how hard the repair was asked to work.  Two train-free
diagnostics now do the direct thing.

Both are pilot-scale in this section (10 sequences x 60 frames from the 60-sequence split, one
severity, one condition `tok_block_erase` at ratio 0.2 on both modalities, one seed); the
60-sequence x 200-frame runs are queued and this section will be replaced by their numbers.

#### 6.28.1 Is the target still in the window when the tracker loses it? (`tools/search_containment.py`)

`Trainer.infer_sequence` records the box that drove each frame's crop, so containment, centre
offset and window-to-object scale ratio are all computable per frame.  Containment alone is a weak
instrument -- the crop square is four object-widths across, so the target stays "inside" it through
a drift of more than one object width -- which is why the other two measures are reported with it.

For the three pilot sequences that fail inside 60 frames:

| quantity (after the failure) | value |
|---|---|
| fraction of the ground-truth box inside the crop | **1.000** (3/3 sequences fully inside) |
| centre offset, in crop sides | **0.205** (0.5 would put the target on the boundary) |
| window side / object side | **5.08** (a correct prediction gives exactly 4.0) |
| frames that are both on-centre and at the right scale | **88.3 %** |

So the target is **in the window, near the centre, at almost the intended scale**, and the tracker
still fails.  And re-running the same condition with the crop schedule replaced by the lagged
ground truth -- which changes *only* where the window points, leaving model, corruption and head
untouched -- lifts mean IoU from **0.235 to 0.801** and SR from 0.272 to 0.978.

The failure is therefore not "the target left the frame": the window is roughly where it should be,
the tracker's *box* is what has drifted, and the drift persists because the next crop is derived
from the drifted box.  That is a within-window self-consistency failure, and the pilot reading is
that **the search range is not the lever**, which is the opposite of what the failure statistics of
6.17.1 suggest on their own.

#### 6.28.2 Is a repaired token set a useful mediator? (`tools/token_oracle_interp.py`)

Every frame is run twice -- once with the token corruption off (caching what fusion receives) and
once with the corruption on, where the corrupted tensors are replaced at the fusion input by
`z_alpha = (1 - alpha) * z_corrupted + alpha * z_clean`.  `alpha = 1` is therefore an **oracle
repair**: the decoder's output is perfect, with the model, the crop loop and the head unchanged.
`alpha = 0` must reproduce the plain corrupted run bit for bit, and it does
(`alpha0_consistent = {fixed: True, free: True}`) -- so the extra clean forward perturbs nothing.

Mean IoU over the 10 pilot sequences:

| mode | `alpha = 0` | `alpha = 0.5` | `alpha = 1` | gain |
|---|---:|---:|---:|---:|
| **fixed crop** (crops from the clean trajectory) | 0.2607 | 0.2714 | 0.2763 | **+0.016** |
| **free loop** (the real tracker) | 0.1823 | 0.2769 | 0.3087 | **+0.126** |

Two things follow, and they are the point of the section.

* **With the crops held fixed, an oracle repair of the tokens barely moves the per-frame response**
  (+0.016 IoU, +1.3 SR points).  On this reading the token damage is not what limits the tracker
  frame by frame.
* **The entire oracle gain appears in the closed loop** (+0.126 IoU, +14 SR points): perfect tokens
  keep the tracker from drifting.  Per sequence the gain is concentrated --
  `boytakingbasketballfollowing` +0.685, `carcominginlight` +0.335, six sequences below +0.07, one
  at -0.013 -- so this is not a uniform improvement but the prevention of catastrophic drift on a
  few sequences.

`--include-taps` (interpolating the backbone FPN taps too, so the shortcut path cannot carry the
corruption the tokens just lost) makes the free-loop gain **smaller**, 0.1823 -> 0.2641, and turns
two sequences negative.  The taps were trained *masked*, so handing them over unmasked is an
out-of-distribution intervention rather than a strictly better repair; the token-only cell is the
cleaner oracle.

#### 6.28.3 What the two together say

Both diagnostics point at the same place, and it is not the tokens:

1. the per-frame response is nearly insensitive to whether the tokens are damaged (6.28.2, fixed
   crop);
2. the target is inside a reasonably framed window when the tracker fails (6.28.1);
3. both the tokens-oracle and the crop-oracle rescue the tracker by preventing or undoing **crop
   drift** (6.28.1, 6.28.2 free loop).

That is a *much* narrower and better supported statement than "per-token repair quality is not the
bottleneck", which 6.27 retracts: what the measurements support is that **the tracking outcome is
governed by the crop trajectory, not by the token-level repair**, on this checkpoint and this
corruption family.  It also re-orders the candidate next steps: a local window re-centring
mechanism is measured to be worth +0.57 IoU on the pilot, while a *global* re-detection stage is
aimed at a failure mode the containment measurement does not show.

Not established here: whether the same holds for image-level conditions (`rgb_occl_04`, the
pre-registered primary) or under closed-loop divergence; whether the crops were already off-centre
*before* the failure; and whether the oracle gains survive at 60 sequences x 200 frames.  All three
are part of the queued run.

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
