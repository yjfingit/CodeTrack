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

---

## 11. Single shared parity-check matrix (post-review revision)

An external review of `b530d9a` found that the first implementation used **three
unrelated connectivity structures** and therefore was not a Tanner decoder in any
meaningful sense:

| was | problem |
|---|---|
| `codebook`: `A : 16x16` (produced the parity) | unrelated to everything downstream |
| `tanner`: `A_uv : 128x128` (variable-variable affinity) | not a parity-check matrix |
| `decoder`: `H : 16x256` (random) | the only real incidence, used by BP alone |

The syndrome, the error locator and the corruption labels were all computed against
structures that the decoder never used.

### 11.1 One `H` now drives the whole chain

`H` lives in the decoder (`decoder.matrix()`, row-normalised, `softplus` so it stays in
`R+`) and is passed explicitly to every stage:

```text
H (16 x 256)
  |-- tanner : check_j = sum_i H[j,i] * v_i          (check nodes, B x 16 x 128)
  |-- syndrome : s_j = D(phi(H v)_j, parity_j)       (visual syndrome, B x 1 x 16)
  |-- loss : y_check = 1[(H @ M) > 0]                (supervision, per modality)
  |-- locator : suspect_i = sum_j H[j,i] * s_j       (H^T, B x 256)
  \-- decoder : m_vc = H v,  m_cv = H^T f_c(...)     (neural BP messages)
```

The check nodes used to be pooled from the 128 graph nodes by a learned slot matrix; they
are now the actual parity checks over all 256 variables. The `128x128` affinity survives
as `A_vv` -- an auxiliary *semantic* graph that only decides which tokens are worth
watching. It is explicitly **not** called a parity-check matrix any more.

### 11.2 Reliability gate that actually gates

The decoder's `residual` was `sum(delta)` **without** the `(1 - r)` factor while the
fusion added `corrected + residual`, so the full un-gated correction was pushed back into
the tracking path even for fully reliable tokens. Now:

* the fusion uses the corrected tokens only;
* the reported residual is `sum((1 - r) * gate * delta)`;
* the gate is computed from `severity` and the **per-modality** reliability gathered onto
  the selected graph nodes, and RGB / TIR get separate gates.

### 11.3 The FPN bypass is gated

`block2` / `block5` taps are taken from the backbone, i.e. *before* token corruption.
Added unconditionally they were a clean shortcut around the entire correction branch.
A learnable `fpn_gate` (initialised with bias `-2`, so `sigmoid ~ 0.12`) now scales the
FPN contribution, and `model.use_fpn: false` removes the path for the ablation.

### 11.4 Per-modality corruption labels

RGB and TIR masks are kept separate, so a healthy modality's reliability is not dragged
down when the other one is damaged. `corruption.target: rgb | tir | both` selects which
modality is damaged -- the setting needed for the "repair the bad modality with the good
one" experiment.

### 11.5 Masked correction

`L_correct` is computed **only on corrupted positions**; a separate `L_preserve` term
(also reported) keeps the decoder from disturbing the ~200 clean tokens. Averaging over all
256 positions let the clean majority dilute the repair objective.

### 11.6 Smaller fixes

* `identity_tokens` now reads `q[:, -16:]` -- the 16 learnable queries, which are
  *appended* after the 128 template tokens. The previous `q[:, :16]` returned template
  tokens while the documentation described query outputs.
* `H` guarantees `min_column_degree = 2`: previously 13.7% of variables (35 / 256) were
  watched by no check at all and could only be "repaired" by the local update MLP.
* The clean teacher no longer runs a second full `_code_path` (it was computed and never
  used); this alone removed ~17% of the step time.
* `identity_map` is a probability (`sigmoid`) rather than an unbounded logit.
* The warm-up stage unfreezes the whole detect+repair chain, not just codebook+decoder --
  otherwise the syndrome and locator stayed random while the loss asked them to localize.
* `timm` is declared in `requirements.txt` / `pyproject.toml`.

### 11.7 The parity tokens now belong to `H`

The deepest residue of the "three unrelated structures" problem was still present after the
first revision: the parity tokens were `P = A·U` with a `16x16` matrix built from the
template, while the check neighbourhoods were defined by a completely unrelated
`H (16x256)`.  The syndrome was therefore comparing a search observation against an
*arbitrary* reference rather than against what that particular check should have seen.

The parity is now **induced by the same incidence**:

```text
w[i, m] = softmax_m ( cos(x_i, U_m) / tau )        # how strongly variable i is
                                                    # explained by codeword m
A_dyn   = H @ w                                      # M x K, the induced parity matrix
parity_j = sum_m A_dyn[j, m] * U_m                  # what check j expects
obs      = H @ x                                     # what check j actually sees
s_j      = D(phi(obs)_j, parity_j)
```

`parity_j` is now, by construction, the value check `j` expects over exactly the
neighbourhood it watches, reconstructed from the **trusted template** rather than from the
possibly corrupted search frame.  The static `16x16` generator (`SparseParityGenerator`)
was removed rather than left as dead code.

`obs` is computed on the **modality-averaged** code-space projection,
`0.5 * (to_code(x_r) + to_code(x_t))`, so damage in either modality enters the syndrome
directly instead of only through the graph construction.

The auxiliary `A_vv` semantic graph is retained but is no longer a parallel "check": it
contributes an additive context term to the check representation, and the primary term is
`obs_proj(H @ observation)`.

### 11.8 Can the claim be checked? Yes -- `tools/diagnostics.py`

`PR / SR / NPR` cannot distinguish "the correction worked" from "the tracker was robust
anyway".  Two additions close that gap.

**`codetrack/engine/evaluator.py`** adds the metrics that *do* speak to the claim:

| metric | definition | chance level |
|---|---|---|
| `syndrome_auroc` | corrupted-frame syndrome vs. **clean-frame** syndrome (the clean pass is the negative class -- without it every check fires and the AUROC is undefined) | 0.5 |
| `syndrome_density_spearman` / `_mae` | rank correlation and MAE between the syndrome and the per-check corruption **density** -- the quantity actually supervised | 0 |
| `locator_precision_at_k` | fraction of the k most suspicious tokens that are corrupted, ranked **within each frame** | the corruption ratio, reported as `locator_precision_chance` |
| `locator_recall_at_k` | corrupted tokens present in the top-k. Reported for continuity, but its **ceiling is `k / n_corrupted`** (~0.098 at k=5, 20% corruption, 256 tokens), so it must be read against that, not against 1.0 | `k / N` |
| `recovery_gain` | `(E_before - E_after) / E_before`, **on corrupted tokens only** | 0 |
| `damage_clean` | error on the untouched tokens, i.e. how much the repair disturbs healthy code | 0 |

Two of these had to be fixed because the metric itself was wrong, not the model:

- `chance_level` returned `k * n_pos / N`, which is the expected **hit count**, not the expected
  recall. At k=5 / 20% corruption / 256 tokens that is 1.77 -- a baseline above 1.0, so a
  perfect localizer scored *worse* than chance. Correct value: `k / N`.
- `recall_at_k` ranked the whole corpus' tokens jointly, so the top-k always came from whichever
  frames were worst. It now ranks within each frame and averages.

**`tools/diagnostics.py`** runs the perturbations that would expose decoration:

| probe | what it breaks | a real ECC model should |
|---|---|---|
| `shuffle-incidence` | permutes the columns of `H` (which variable each check watches) | degrade sharply |
| `shuffle-parity` | permutes the parity order | degrade |
| `shuffle-syndrome` | permutes the syndrome entries | degrade |
| `no-decoder` | bypasses the BP decoder entirely | tracking collapses, `recovery_gain` becomes exactly 0 |

Run it with:

```bash
python tools/diagnostics.py --checkpoint outputs/<exp>/final.pth --sequences 5
```

The rule is simple and deliberately blunt: **a probe that does not hurt the metrics is
decoration, not correction.**

### 11.9 Architecture-property tests

`tests/unit/test_architecture.py` locks one invariant per review finding -- identity-token
permutation invariance (P0-6), a single row-normalised non-negative incidence (P0-1),
minimum column degree (P1-4), `locator_scattered == s @ H` (P0-3), unit-interval gates
(P1-6), independent per-modality corruption masks (P0-7), a parameter-matched MLP
baseline, a removable FPN path (P0-4) and a fully frozen backbone.  They run on a 2-block
dim-64 model, so the whole file finishes in seconds on CPU.

> These tests earned their keep immediately: writing them surfaced three latent bugs that
> the default configuration had been hiding -- `node_to_variable` had `128` hard-coded, the
> minimum-degree repair indexed the wrong tensor dimension, and the MLP baseline still
> carried the (now unused) BP message layers.

---

## 12. Shortcut closures (second review round)

Three structures were letting the tracker take an easier path than the one the architecture
claims. Each is now closed *structurally*, i.e. the shortcut no longer exists in the graph,
rather than by a weight that merely discourages it.

### 12.1 Syndrome supervision: OR target -> density target

```python
# before -- degenerates at this scale
check_target = ((h @ mask_any.t()) > 0).float().t()
# after
degree  = H_support.sum(dim=1).clamp(min=1.0)
density = ((H_support @ mask_any.t()) / degree.unsqueeze(1)).t().clamp(0, 1)
```

`min_column_degree >= 2` over `N = 256` forces `>= 2*256/16 = 32` edges per check, so under 20%
erasure `1[(H@M) > 0]` is ~1 with probability 0.999. The BCE had degenerated into "predict 1
everywhere". The density target uses the **fixed binary support**, never the learnable weights:
the neighbourhood a check watches is structural and must not be allowed to move to make the
target easier.

### 12.2 Tanner: the parity reference no longer appears on both sides

```python
# before -- the syndrome was compared against a tensor that already contained the parity
checks = self.check_norm(obs + sem + self.parity_to_check(parity))
# after
sem = sem * torch.sigmoid(self.sem_gate)
checks = self.check_norm(obs + sem)
```

`VisualSyndrome` already computes `D(phi(checks), parity_ref(parity))`. With `parity` also
summed into `checks`, the learned discrepancy could match the two copies of the same tensor
instead of testing whether `H x` actually agrees with the code. The `parity_to_check` layer is
**deleted**, not merely disconnected -- leaving it would be 33 K dead parameters.

Two further guards on the reference itself:

| guard | before | after |
|---|---|---|
| assignment temperature `tau` | `abs().clamp(min=1e-2)` on a raw parameter: can reach 0 (all variables collapse onto one identity codeword) or explode (all assigned equally, parity stops being a weighted recombination) | `tau = tau_min + (tau_max - tau_min) * sigmoid(rho)`, learnable, bounded in `[0.02, 0.5]` |
| who decides the mixture `w[i,m]` | `cos(x_i, U_m)` with `x_i` the **corrupted** search token -- the redundancy was steered by the word it must repair | the token is first shrunk towards its neighbourhood mean by its own reliability: `v <- normalize(r*v + (1-r)*mean(v))`, so a distrusted token cannot move `w` |

### 12.3 FPN: the bypass is masked, not gated

The block-2 / block-5 taps are read from inside the backbone, i.e. **before** token corruption.
`fpn_gate` (bias -2, sigmoid ~0.12) only slowed the shortcut down; `L_track` could always open
it, because the clean answer was sitting right there. `CodeTrack._apply_tap_corruption` now
applies the *same* corruption mask at the *same* 16x16 positions to the search part of every
tap, before the FPN projection:

```python
z_part, x_part = feat[:, :lens_z], feat[:, lens_z:]
x_part = x_part * (1.0 - mask).unsqueeze(-1)
```

Template tokens are untouched. Whatever the decoder fails to repair is now also missing from
the FPN path, so the two cannot be compared on unequal information.

### 12.4 `L_correct`: three sub-terms instead of one

| sub-term | closes |
|---|---|
| `L_preserve` (0.5) | a decoder that "repairs" by rewriting *every* token scores the same as one that fixes the damaged ones -- the healthy tokens are now pinned too |
| `L_gain` (1.0, beta = 0.9) | returning the corrupted input unchanged: `e_after = e_before` gives margin `(1-beta) * e_before > 0` |
| `L_correct` (1.0) | the original absolute repair term |

This also settles the "26 M cannot restore an 86 M encoder" worry. The decoder never
re-simulates the backbone: the teacher is the **same frozen backbone's** clean tokens, and the
corruption is applied to feature tokens *after* the backbone. The task is feature-space
denoising, and `L_gain` is what forces the decoder to actually use the redundancy instead of
learning the identity.

### 12.5 Naming

The decoder aggregates messages per node over the whole `H`; it has no per-edge extrinsic
exclusion, so it is not textbook BP. The code, config comments and `docs/method.md` now say
**neural Tanner decoding / BP-style**. The claim that is actually owned -- and that
`tools/diagnostics.py` exists to defend -- is that one shared `H` drives check aggregation, the
supervision label, the `H^T` localization vote and the message passing, so those four cannot
drift apart. `tests/unit/test_architecture.py` grew from 16 to 26 tests: density target (x2),
no parity in `checks`, bounded `tau`, reliability-gated `w`, masked FPN taps, `L_gain` biting,
`L_preserve` applied to the total, `chance_level` as a recall, per-frame recall ranking.
