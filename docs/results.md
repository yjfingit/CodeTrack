# Results

> Placeholder. Every entry must carry the config path, commit hash, seed and log path.
> Rule: no number without a reproducible command (see `docs/reproducibility.md`).

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
