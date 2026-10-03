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
| Losses (`L_track`, `L_detect`, `L_correct`, `L_identity`) | `codetrack/engine/losses.py` |

---

## 10. Implementation notes (how the figure maps onto this repository)

Every place where the figure leaves room for interpretation, and what the code does.

### 10.1 Backbone: OSTrack ViT-B/16 with absolute position embeddings

The figure annotates the stack as `12 Blocks, Frozen, RoPE, Abs. Pos`. Because the
requirement is to **reuse OSTrack's public weights directly**, the implementation keeps
OSTrack's absolute position embeddings and does **not** use RoPE -- re-deriving RoPE
would invalidate the pretrained weights. Attention, MLP blocks, patch embedding and
position embeddings are ported from OSTrack (`lib/models/ostrack/vit.py`, Apache-2.0)
and stay bit-compatible, so `OSTrack_ep0300.pth.tar` loads with no key surgery
(**151 tensors**, verified; the search/template position grids come from the checkpoint
itself, so they are equally exact).

* RGB keeps the pretrained 3-channel patch embedding.
* TIR gets a **separate 1-channel** patch embedding initialised as the channel-mean of
  the RGB filter -- a thermal frame is approximately the luminance of the visible frame.
  The figure draws one `Patch Embed` box per modality, which this matches.
* The 12 transformer blocks are **shared** across modalities and frozen
  (`freeze_backbone: true`).
* If a checkpoint has no `pos_embed_z` / `pos_embed_x`, they are re-derived by bicubic
  interpolation from `pos_embed`, exactly as OSTrack's `finetune_track` does.

### 10.2 Why `A_uv` is `128 x 128`

The selector keeps the best **128 of the 256** variable nodes as graph nodes, so their
pairwise affinity matrix is `128 x 128` with the 16 parity tokens as 16 check nodes
(`16 x 128` after embedding). `Softmax_s` is a **sparsified softmax**: only the top-8
affinities per row survive before renormalisation. That sparsity is a functional
requirement, not a speed trick -- it is what lets a failing-check pattern vote for one
specific token during localization.

### 10.3 Why FPN is 512 per modality while the head still sees 768

Block-2 and block-5 taps are each projected to 256 channels and concatenated, giving
exactly the figure's `B x 512 x 16 x 16` per modality. The two modalities are then
fused back to **768** so the tracking head can reuse OSTrack's head weights
(90/92 tensors; the two skipped entries are the fixed coordinate grids, which are
buffers rather than trained weights).

### 10.4 The correction path is dual-modality over one shared graph

`Corrected zg` and `Corrected xg` in the figure are the RGB and TIR tokens repaired by
the **same** parity-check matrix `H` (`16 x 256`, sparse) driven by the **same**
syndrome. Both modalities are stacked along the batch dimension, so the two repairs
cost a single fused forward pass.

### 10.5 Verified tensor shapes

`python tools/check_shapes.py --ckpt checkpoints/pretrained/OSTrack_ep0300.pth.tar`
audits a forward pass against the figure and currently reports **21/21 tensors OK**,
from `z_r: (1, 64, 768)` through `bbox: (1, 4)`.

### 10.6 Parameters

| module | trainable / total |
|---|---|
| backbone (frozen) | 0.003 M / 86.244 M |
| codebook | 5.983 M |
| reliability + selector | 1.578 M |
| tanner | 0.495 M |
| syndrome + locator + gating | 0.163 M |
| decoder (BP) | 4.928 M |
| fusion | 6.885 M |
| head | 6.474 M |
| **total** | **26.51 M trainable / 112.75 M** |
