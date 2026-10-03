# CodeTrack Architecture

> Tensor shapes in this document follow the reference configuration:
> ViT-B/16 backbone, `M = 16` parity tokens, `K = 16` identity tokens, `N = 256` variable nodes,
> RGB search frame `B x 3 x 256 x 256`, TIR search frame `B x 1 x 256 x 256`,
> RGB/T template `B x 3 x 128 x 128`.

## 0. Data flow (one-line)

```text
RGB/T Template (+ Trusted Memory)
      -> Target Codebook Encoder
      -> {Identity Tokens U, Sparse Parity Tokens P}
RGB/T Search -> Shared ViT-B/16 -> target candidate tokens V
      -> Reliability Estimator
      -> Reliability-Adaptive Tanner Graph
      -> Visual Syndrome Computation (S)
      -> Error Locator / Severity
      -> Neural BP Decoder (x2)
      -> Corrected tokens + Residual
      -> Fusion -> Tracking Head -> Bounding Box
```

## 1. Backbone — `codetrack/models/backbone/`

Kept intentionally standard; no contribution is claimed here.

- Patch embed `16x16`, stride `16`
- Shared ViT-B/16, 12 transformer blocks, `pretrained = off`
- `rgb_search`: `B x 3 x 256 x 256` -> patch embed `B x 256 x 768` -> tokens `B x 320 x 768`
- `tir_search`: `B x 1 x 256 x 256` -> patch embed `B x 256 x 768` -> tokens `B x 320 x 768`
- Template branch output: `B x 64 x 768`
- Block 2 / block 5 outputs are retained for FPN fusion (`B x 256 x 768`, `B x 256 x 768`)

## 2. Target Codebook Encoder — `codetrack/models/codebook/`

Inputs: `(z_R, z_T, M_{t-1})` with `z_R, z_T: B x 64 x 768`.

- Concatenate template tokens -> `B x 128 x 768`
- Project to codebook input -> `B x 128 x 512`
- Linear to query -> `B x 144 x 768`
- Split into:
  - **Identity Tokens** `U`: `B x 16 x 256`
  - **Sparse Parity Tokens** `P`: `B x 16 x 256` (via ECC projection, `768 -> 256`)
- Parity is generated from *trusted* information only (initial template + reliable history),
  never from the possibly corrupted current search frame:
  `P = A_theta U`, `A` sparse with `A_ij in {0,1}` or soft `A_ij in [0,1]`.

## 3. Reliability Estimator — `codetrack/models/reliability/`

- Query: `B x 256 x 768` (RGB), `B x 64 x 768` (TIR)
- Output: token reliability `r_i in [0,1]` from template similarity, RGB-T consistency and
  temporal consistency.
- **Target Candidate Selector**: `N = 256` variable nodes selected from `B x 256 x 768` RGB/T tokens,
  forming `V = {v_R^1..v_R^N, v_T^1..v_T^N}` (`B x 256 x 768` after selection).

> Reliability alone is *not* the contribution; it is an input to decoding.

## 4. Reliability-Adaptive Tanner Graph — `codetrack/models/tanner/`

- **Dynamic Graph Construction**: `A_uv = softmax_s(U U^T / sqrt(d))`, shape `B x 128 x 128`
- Variable nodes: `N x (2N)` connectivity `B x 256 x 256` (i.e. `256 x 512` incidence)
- Check nodes: `M` parity checks `B x 16 x 128`
- Incidence / identity map `B x 128 x 256`
- Node degree is not uniform (see UEP in `docs/method.md`): important tokens connect to more checks.

## 5. Visual Syndrome Computation — `codetrack/models/syndrome/`

```text
s_j = D( phi({v_i : i in N(j)}), p_j )
```

- `phi` = sparse aggregation over the check neighbourhood
- `D` = `L1` / `L2` / learned discrepancy
- Output `S`: `B x 1 x 16` — a **Visual Syndrome Map**, not an attention map
- Healthy token: `s_j ~ 0`;  corrupted token: `s_j >> 0`
- **Error Locator** uses sparsity: if checks `{2,7,9}` all fire and all connect `v_R^17`,
  then `v_R^17` is the prime suspect.
- **Error Severity / Reliability-Aware Gating**: `B x 1 x 128`

## 6. Neural BP Decoder — `codetrack/models/decoder/`

Two iterations of learned belief propagation:

```text
Variable -> Check:  m_{i->j}^{(l)} = f_v( v_i^{(l)}, r_i, m_{k->i}^{(l-1)} )
Check -> Variable:  m_{j->i}^{(l)} = f_c( s_j, p_j, {m_{k->j}} )
Update:             v_i^{(l+1)} = v_i^{(l)} + (1 - r_i) * dv_i^{(l)}
```

- Corrected variable nodes: `B x 256 x 768`
- Residuals: `B x 256 x 768`
- High-reliability tokens barely move; low-reliability tokens absorb correction.
- `(1 - r_i)` gating is what allows reliable RGB to repair TIR *and* vice versa, while
  template/history parity prevents both modalities drifting together.

## 7. Fusion + Tracking Head — `codetrack/models/fusion/`, `codetrack/models/head/`

- `Corrected zg` `B x 256 x 768`, `Corrected xg` `B x 256 x 768`
- Residual add -> `B x 256 x 768`
- Reshape to feature map -> `B x 768 x 16 x 16`
- FPN fusion between block 2 / block 5 features -> `B x 768 x 16 x 16`
- Tracking head output: `B x 4 x 16 x 16` -> bounding box

## 8. Training corruption — `codetrack/data/corruption/`

| Channel | Perturbation |
|---|---|
| RGB | low-light, over-exposure, blur, occlusion, color degradation |
| TIR | thermal saturation, thermal crossover, noise, contrast loss |
| Token | random erasure, burst erasure, feature noise |
| Cross-modal | spatial shift, scale shift, temporal delay |

Corruption positions are known at training time, enabling a supervised syndrome loss.

## 9. Module -> directory map

| Architecture block | Directory |
|---|---|
| Shared ViT-B/16, patch embed | `codetrack/models/backbone/` |
| Target Codebook Encoder | `codetrack/models/codebook/` |
| Reliability Estimator, Target Candidate Selector | `codetrack/models/reliability/` |
| Reliability-Adaptive Tanner Graph | `codetrack/models/tanner/` |
| Visual Syndrome Computation, Error Locator | `codetrack/models/syndrome/` |
| Neural BP Decoder | `codetrack/models/decoder/` |
| Fusion, residual, reshape | `codetrack/models/fusion/` |
| Tracking head | `codetrack/models/head/` |
| Losses (`L_track`, `L_corr`, `L_syn`, `L_parity`) | `codetrack/engine/losses.py` |
