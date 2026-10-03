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
checks_j = LN( (H x)_j + g * sem_j )          # observation side: H, no parity
s_j      = D( phi(checks_j), parity_ref(p_j) ) # compared against the trusted reference
S        = [s_1 ... s_M]                       # Visual Syndrome Map
```

- **Detection**: `S -> P(corruption)`.
- **Localization**: graph sparsity lets failing checks vote for a suspect variable. If checks
  `{2,7,9}` fire and all connect `v_R^17`, that token is the prime suspect. This is the capability
  plain uncertainty fusion cannot provide.

### 6.1 The syndrome is trained on a *density*, not on a bit

The supervision target for check `j` is the **fraction of its neighbourhood that is corrupted**

```text
q_j = sum_i 1[H_ji > 0] * M_i  /  sum_i 1[H_ji > 0]
```

not the binary `1[ (H M)_j > 0 ]`. The binary version is degenerate at this scale. Requiring
`min_column_degree >= 2` over `N = 256` variables forces at least `2 * 256 / 16 = 32` edges per
check, so a check that watches 32 tokens sees a 20%-corrupted frame with probability
`1 - 0.8^32 ~ 0.999`, and the BCE degenerates into "predict 1 everywhere" -- it carries no
gradient signal about *where* the damage is. The density stays informative at any corruption
ratio and lets training keep the hard 20% setting rather than lowering it to make the syndrome
learnable. `1[(H M) > 0]` is kept only as a coarse evaluation label.

### 6.2 The reference must not appear on both sides

Two shortcuts would make the syndrome uninformative without any real coding happening, and both
are closed structurally:

| Shortcut | Why it is a shortcut | What the code does |
|---|---|---|
| `checks = LN( H x + sem + parity_to_check(p) )` while the syndrome compares `checks` against `parity_ref(p)` | the discrepancy is computed against a tensor that already contains `p`; the network can match the two trivially instead of testing whether `H x` really agrees with `p` | the parity projection is **removed** from the check construction. `checks` is built from the observation only; `p` appears exclusively on the reference side. |
| `w[i,m] = softmax(cos(x_i, U_m) / tau)` computed from the corrupted search tokens | the mixture coefficients of the "redundancy" are decided by the very word the redundancy is supposed to repair | `w` is gated by the per-token reliability: a token the model already distrusts is shrunk towards its neighbourhood mean before the cosine, so it cannot steer `p`. |

The assignment temperature is `tau = tau_min + (tau_max - tau_min) * sigmoid(rho)`, learnable
yet bounded in `[0.02, 0.5]`. A raw `abs().clamp(min=1e-2)` parameter can sit at 0 (every
variable collapses onto one identity codeword) or explode (every variable assigned equally, so
`p` stops being a weighted recombination of the codebook); both extremes silently destroy the
coding meaning of the parity without changing any shape.

## 7. Neural Tanner decoding (BP-style)

```text
Variable -> Check:  m_{i->j}^{(l)} = f_v( v_i^{(l)}, r_i, m_{k->i}^{(l-1)} )
Check -> Variable:  m_{j->i}^{(l)} = f_c( s_j, p_j, {m_{k->j}} )
Update:             v_i^{(l+1)} = v_i^{(l)} + (1 - r_i) * dv_i^{(l)}
```

### 7.1 What this decoder is, and is not

The messages are aggregated **per node over the whole `H`**: every variable produces one
message, `H` pools it into 16 checks, and `H^T` broadcasts the result back. Genuine BP's
`m_{i->j}` / `m_{j->i}` are *edge-specific extrinsic* messages that exclude the message just
received on that same edge; this implementation has no such exclusion.

So the claim is **neural Tanner decoding / syndrome-guided neural error correction**, and the
paper should say *BP-style*, not *exact BP*. What the implementation does own is that a single
`H` drives check aggregation, the supervision label, the `H^T` localization vote and the
decoder's message passing -- so the four mechanisms cannot drift apart. `tools/diagnostics.py`
is what has to convince a reviewer: `shuffle-incidence`, `shuffle-parity`,
`shuffle-syndrome` and `no-decoder` must each visibly degrade `recovery_gain` and
localization. A probe that does not hurt anything is decoration.

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
| `L_detect` | BCE on reliability vs. the known token-corruption mask; **soft-BCE on the syndrome vs. the per-check corruption density**; BCE on the locator vs. the per-token mask | without it the syndrome can collapse to a constant, and "detection" carries no information. **The model must find the corruption, not merely survive it.** The density target (see 6.1) is what keeps the term from degenerating. |
| `L_correct` | three sub-terms, weight **2.0** in total | this is the core claim: the decoder must actually restore the corrupted feature. It is supervised directly against the clean tokens rather than only through the tracking loss. |
| `L_identity` | distillation over the identity codebook: the soft assignment of a repaired token over `U` must match that of its clean counterpart, weight 0.1 | stops the decoder from "repairing" a token onto a **look-alike distractor** (another person, a similar vehicle) instead of the tracked target |

### The three sub-terms of `L_correct`

An absolute distance to the clean teacher is minimised by *any* small output delta, including a
delta that helps not at all. Two failure modes have to be closed explicitly:

```text
L_correct  = mean_{i in corrupted}  || x~_i - x*_i ||_1                  # repair the broken ones
L_preserve = mean_{i in healthy  }  || x~_i - x*_i ||_1                  # leave the rest alone
L_gain     = relu( e_after - beta * e_before ),   beta = 0.9             # beat your own input
e_before   = mean_{i in corrupted} || x~_i - x*_i ||_1   (the corrupted input)
e_after    = mean_{i in corrupted} || x~_i^dec - x*_i ||_1 (the decoder's output)
total      = L_correct + 0.5 * L_preserve + 1.0 * L_gain
```

- `L_preserve` alone: a decoder that "repairs" by rewriting **every** token scores exactly as
  well on `L_correct` as one that fixes the damaged ones and leaves the healthy ones intact.
- `L_gain` alone: with `beta < 1`, returning the corrupted input unchanged is *not* a solution
  -- it yields `e_after = e_before`, so the margin is `(1 - beta) * e_before > 0`. Note this is
  feature-space denoising, not "26 M parameters re-simulating an 86 M encoder": the teacher is
  the *same* frozen backbone's clean tokens, one pass away.

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
