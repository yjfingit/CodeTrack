# Corruption Protocol

Goal: make robustness claims **reproducible** and comparable across trackers.

## 1. Corruption taxonomy

| Level | ID | Corruption | Applied to |
|---|---|---|---|
| Image / RGB | `rgb_lowlight` | gamma-darkened low light | RGB frame |
| Image / RGB | `rgb_overexp` | over-exposure / clipping | RGB frame |
| Image / RGB | `rgb_blur` | Gaussian / motion blur | RGB frame |
| Image / RGB | `rgb_occl` | rectangular occlusion | RGB frame |
| Image / RGB | `rgb_color` | color degradation / desaturation | RGB frame |
| Image / TIR | `tir_sat` | thermal saturation | TIR frame |
| Image / TIR | `tir_cross` | thermal crossover (target/background contrast loss) | TIR frame |
| Image / TIR | `tir_noise` | sensor noise | TIR frame |
| Image / TIR | `tir_contrast` | contrast compression | TIR frame |
| Token | `tok_random_erase` | random token erasure | feature map |
| Token | `tok_burst_erase` | contiguous flattened burst erasure | feature map |
| Token | `tok_block_erase` | rectangular 2-D block erasure | feature map |
| Token | `tok_feat_noise` | additive feature noise | feature map |
| Cross-modal | `xmodal_shift` | spatial shift between modalities | pair |
| Cross-modal | `xmodal_scale` | scale mismatch | pair |
| Cross-modal | `xmodal_delay` | temporal delay / desync | pair |

## 2. Severity levels

Every corruption is parameterized by a single scalar severity in `{0.1, 0.2, 0.4, 0.6}` plus a
corruption ratio for the stochastic ones (`0.2`, `0.4`).

| Severity | Meaning |
|---|---|
| `mild` | `0.1` |
| `moderate` | `0.2` |
| `severe` | `0.4` |
| `extreme` | `0.6` |

## 3. Evaluation matrix

```text
datasets   : LasHeR, RGBT234
protocols  : RGB-only corrupt, TIR-only corrupt, both corrupt
severities : mild / moderate / severe / extreme
scenarios  : random, burst, erasure, misalignment
metrics    : PR, SR (AUC), NPR
```

Configs are generated into `configs/corruption/`; one yaml per (corruption, severity) cell.

## 3.1 Initial calibration protocol

Calibrate perturbation strength on held-out validation sequences with one frozen no-decoder
checkpoint before comparing trained methods. Use the same sequence order and corruption seed
for every cell. Report the full clean-to-corrupt curve; do not choose a single level because
one method happens to look best there.

| Test | Initial levels | Measurement detail |
|---|---|---|
| 2-D token block erase | corrupted token ratios 0.1, 0.2, 0.3, 0.4 | one to three rectangles; report realized mask ratio |
| Random token erase | match each block mask's corrupted-token count | spatial-structure control |
| RGB image occlusion | target coverage about 20%, 40%, 60% | compute overlap on the target crop, not the whole search image |
| RGB low light | brightness multipliers 0.7, 0.4, 0.2 | keep TIR unchanged; record pixel range and noise level |
| TIR contrast loss | retain 70%, 40%, 20% of target/background contrast | proxy only; verify against annotated thermal-crossing sequences |

Run RGB-only, TIR-only, and both-modality corruption separately. For temporal robustness,
keep each event active for 5, 15, and 30 frames, then include an uncorrupted recovery period.
These starting levels are an engineering calibration, not a universal definition of realistic
damage.

For feature recovery, use target-aligned template and search crops and keep RGB and TIR masks
independent. Report damaged-region error, healthy-region damage, and target-region error in
addition to tracking metrics. `tools/recovery_probe.py` currently implements target-aligned
feature-level probes; image-area and temporal-event calibration require their own evaluation
cells.

## 4. Reproducibility requirements

1. Corruption is applied on-line, **after** the deterministic data pipeline, with a fixed seed
   recorded in the run manifest.
2. Corruption position masks are exported alongside predictions (`M_err`), enabling the syndrome
   localization metric.
3. Baseline trackers must be evaluated with the identical corruption stream (same seed, same order).

## 5. Extra metrics for error correction

| Metric | Definition |
|---|---|
| Detection AUROC | `S` vs. ground-truth corruption indicator |
| Localization recall | fraction of corrupted tokens found in the top-k suspect set |
| Recovery delta | `SR_corrupt(after correction) - SR_corrupt(no correction)` |

## 6. Validation split and paired statistics (implemented)

`tools/select_validation_sequences.py` builds a seeded, stratified split from LasHeR's own
attribute annotations (`AttriSeqsTxt/`), so the validation set is not "the first N lines of
`testingsetList.txt`". Strata are assigned in a fixed priority order and recorded in
`manifest.json`: `low_illumination` (`LI`), `thermal_crossover` (`TC`), `total_occlusion`
(`TO`), `partial_occlusion` (`PO` without `NO`), `unoccluded`.

The current split `outputs/validation_split_v1/` holds 60 `testingset` sequences, 12 per
stratum, 200-frame segments, seed 0. Every condition runs on that same list through
`tools/validate_sequences.sh`, which preserves the per-process thread caps and writes the
exact commands and environment into `run_manifest.json`.

| condition | flags |
|---|---|
| `clean` | no corruption |
| `tok_block_rgb_02`, `tok_block_tir_02`, `tok_block_both_02` | `tok_block_erase`, ratio 0.2, target `rgb` / `tir` / `both` |
| `tok_block_rgb_04` | `tok_block_erase`, ratio 0.4, target `rgb` |
| `tok_random_rgb_02` | `tok_random_erase`, ratio 0.2, target `rgb` (spatial-structure control) |
| `rgb_lowlight_04`, `rgb_occl_04` | image-level RGB, severity 0.4 |
| `tir_crossover_04` | image-level TIR, severity 0.4 |

Image-level cells carry no token mask, so their token-level locator/recovery numbers are
reported as `NaN` rather than a fabricated zero; their syndrome AUROC is computed against the
pristine frame, not against the degraded one.

Statistics (`tools/summarize_paired_conditions.py`): **SR is the pre-registered primary
metric**. Each condition is paired with the same run's `clean` value sequence by sequence. The
interval resamples *sequences*, never frames, because adjacent frames are not independent;
paired t and Wilcoxon signed-rank p-values are reported with a Holm correction across the
conditions of a run. Two checkpoints are compared with a paired sequence-level difference on
the identical list and corruption stream.

Two secondary views are reported next to the primary one, because they were needed as soon as
the clean baseline was measured:

* **reference-tracking subset** -- a sequence the reference arm never tracked (clean SR
  <= 0.2) has no room to get worse, so the full-set delta understates the corruption effect
  on sequences that are actually being followed. The same paired statistics are recomputed on
  the subset the reference tracks. The full set stays the primary comparison.
* **loss time** (`tools/horizon_probe.py`) -- per-sequence IoU trace, the first frame after
  which the 20-frame rolling mean IoU never returns above 0.5, and the resulting distribution
  of loss times. SR alone cannot distinguish "tracks at 0.45 for 200 frames" from "tracks
  perfectly for 60 frames and then drifts away", and corruption can only change the loss time
  of sequences that were being tracked at all.

Pre-registered strength screen (an engineering screen, not a general law): the frozen
no-decoder checkpoint should lose roughly 20-40 % relative SR against clean at the reference
level while keeping about 60-80 % of its clean SR; the drop must show a paired sequence-level
interval that excludes zero; and the whole curve is reported rather than the single level that
happens to favour one arm.
