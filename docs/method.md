# Method

## 1. Motivation

Existing RGB-T trackers handle modality degradation by *suppressing* unreliable information
(uncertainty fusion / quality weighting) or by *reconstructing* a missing modality. CodeTrack takes a
third stance: degradation is a **structured corruption of a multimodal target code**, and can be
**detected, localized and corrected** the way a channel decoder repairs a noisy codeword.

```text
Fusion  ->  Verify  ->  Diagnose  ->  Correct  ->  Fuse
```

## 2. Coding-theoretic mapping

| Coding concept | Tracker counterpart |
|---|---|
| variable node | visual token `v_i` of the search frame |
| check node | parity token `p_j` from the target codebook |
| parity check matrix `A` | learnable sparse adjacency (Tanner graph) |
| syndrome `s_j` | per-check discrepancy `D(phi(v_{i in N(j)}), p_j)` |
| belief propagation | iterative neural message passing over the graph |
| soft reliability | token confidence `r_i` |

## 3. Where redundancy must come from

Generating parity from the *already corrupted* frame carries no coding power: `P = f(X~)` inherits the
corruption. Redundancy must originate from **independent**, trusted sources. Single object tracking
naturally provides three:

1. the initial template,
2. the other modality,
3. reliable historical states `M_{t-1}`.

Hence the codebook is generated from `{Z_R, Z_T, M_{t-1}}`, never from the current search frame.

## 4. Target codebook

```text
U_t = E_c(Z_R, Z_T, M_{t-1})        # K identity tokens
P_t = G_theta(U_t) = A_theta U_t    # M parity tokens, A sparse or soft-sparse
```

Each parity token connects to only a few identity tokens, forming a learnable Tanner graph.

## 5. Checks: cross-modal, temporal, spatial

| Check | Form | Targets |
|---|---|---|
| Cross-modal | `p_j^{RT} = g(u_R^i, u_T^j)` | single-modality degradation (e.g. RGB over-exposure) |
| Temporal | `p_j^{time} = g(u^{t-1}, u^{t-2}, z_0)` | tracker drift onto a background distractor |
| Spatial group | `p_j^{spa} = g(u_{i1}, u_{i2}, u_{i3})` | local occlusion / part corruption |

## 6. Visual syndrome

```text
s_j = D( phi({v_i : i in N(j)}), p_j )
S   = [s_1 ... s_M]        # Visual Syndrome Map
```

- **Detection**: `S -> P(corruption)`.
- **Localization**: graph sparsity lets failing checks vote for a suspect variable. If checks
  `{2,7,9}` fire and all connect `v_R^17`, that token is the prime suspect. This is the capability
  plain uncertainty fusion cannot provide.

## 7. Neural belief-propagation decoding

```text
Variable -> Check:  m_{i->j}^{(l)} = f_v( v_i^{(l)}, r_i, m_{k->i}^{(l-1)} )
Check -> Variable:  m_{j->i}^{(l)} = f_c( s_j, p_j, {m_{k->j}} )
Update:             v_i^{(l+1)} = v_i^{(l)} + (1 - r_i) * dv_i^{(l)}
```

Two to three iterations suffice. The `(1 - r_i)` gate means reliable RGB can repair TIR and reliable
TIR can repair RGB, while template/history parity stops both from drifting together.

## 8. Unequal error protection (UEP)

Tracking importance is non-uniform: target centre >> background; discriminative template tokens >>
ordinary ones.

```text
q_i = semantic importance
d_i = f(q_i, r_i)          # number of parity checks attached to variable i
important: d_i = 5 ;  ordinary: d_i = 2 ;  background: d_i = 0
```

Improves robustness and bounds computation at the same time.

## 9. Rateless / adaptive redundancy (stage 2)

```text
RGB ok, TIR ok      -> M = 4
RGB degraded        -> M = 12
RGB + TIR degraded  -> M = 24
```

Opposite of early-exit trackers: spend *more* error-correction budget as corruption grows.

## 10. Training losses

Four terms, with the reference weights ``lambda_d = 1.0``, ``lambda_c = 2.0``,
``lambda_i = 0.1``::

    L = L_track + lambda_d * L_detect + lambda_c * L_correct + lambda_i * L_identity

| Term | Supervision | Why it is needed |
|---|---|---|
| `L_track` | focal (centre heatmap) + L1 + GIoU on the predicted box | keeps the tracker itself learning while the correction machinery is trained |
| `L_detect` | BCE on reliability vs. the known token-corruption mask; BCE on the syndrome vs. per-check corruption; BCE on the locator vs. the per-token mask | without it the syndrome can collapse to a constant, and "detection" carries no information. **The model must find the corruption, not merely survive it.** |
| `L_correct` | `L1(corrected_tokens, clean_teacher_tokens)` for both modalities, weight **2.0** | this is the core claim: the BP decoder must actually restore the corrupted feature. It is supervised directly against the clean tokens rather than only through the tracking loss. |
| `L_identity` | distillation over the identity codebook: the soft assignment of a repaired token over `U` must match that of its clean counterpart, weight 0.1 | stops the decoder from "repairing" a token onto a **look-alike distractor** (another person, a similar vehicle) instead of the tracked target |

### The clean teacher

`L_correct` and `L_identity` need a corruption-free reference. It is obtained from the
**same forward pass**: the frozen backbone is executed once and its tokens are branched
into a corrupted and a clean path; only the CodeTrack modules are run twice, and the
clean branch is wrapped in `no_grad`. Measured overhead on the reference
configuration: none on the backbone, roughly one extra pass over the ~26 M trainable
parameters.

### Staged training

`train.codec_warmup_epochs` (default 1) freezes everything except the codebook and the
BP decoder for the first epochs, so detection + repair are learned before the rest of
the network starts adapting. Afterwards everything except the backbone is unfrozen for
joint fine-tuning.

### Minimal viability configuration

`configs/experiment/lasher_vitb_minimal.yaml` disables `L_identity`
(`lambda_identity: 0.0`) so the first experiment answers one question only: can the
decoder recover corrupted target tokens (`L_correct` falls) and does the model localize
them (`L_detect` falls)?

## 11. Relation to prior art

| Work | Difference |
|---|---|
| SCDT (missing-modality reconstruction) | generative/denoising recovery vs. constraint-driven, diagnosable structured correction |
| DSRTrack (degraded online state recovery) | state repair vs. latent representation correction; discrepancy-based vs. parity-check-based diagnosis; repair module vs. explicit Tanner graph; one-shot/adaptive vs. iterative belief propagation |
| NNCL (neural network coding layer) | generic classification backbones, no cross-modal / temporal / template redundancy, no syndrome localization |

## 12. Contributions (three)

1. **Problem** — robust RGB-T tracking as error correction over multimodal target representations.
2. **Method** — reliability-adaptive Tanner graph with cross-modal, temporal and template redundancy,
   decoded by syndrome-guided iterative message passing.
3. **Protocol** — systematic evaluation under random / burst / erasure / misalignment corruption.
