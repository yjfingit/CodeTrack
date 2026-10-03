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

```text
L_corr  = || F_hat - F_clean ||_1
L_syn   = BCE(M_hat_err, M_err)            # corruption positions known at train time
L_track = L_cls + lambda1 * L_L1 + lambda2 * L_GIoU
L       = L_track + lambda_c * L_corr + lambda_s * L_syn + lambda_p * L_parity
```

A clean teacher supplies `F_clean`; corrupt views supply `F_corrupt`.

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
