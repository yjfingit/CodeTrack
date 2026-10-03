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
| Token | `tok_burst_erase` | contiguous burst erasure | feature map |
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
