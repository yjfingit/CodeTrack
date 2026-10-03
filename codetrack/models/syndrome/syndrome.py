"""Visual Syndrome, Error Locator and severity gating -- architecture figure block 5.

    SP -> SP -> Syndrome S                B x 1 x 16
    Error Locator                         B x 128 x 256
    Error Severity (Reliability-Aware Gating)   B x 128

The syndrome is the conceptual core: for a healthy code it is ~0, for a corrupted
token the checks that watch it light up.  Two capabilities follow from it:

* **detection** -- ``S -> P(corruption)``;
* **localization** -- because ``A`` is sparse, the *pattern* of failing checks votes
  for a specific variable node.  Plain uncertainty fusion cannot do this.

Output parameterisation
-----------------------
``S`` is supervised with a **soft target** (see ``engine/losses.py``): the corruption
*density* inside each check's neighbourhood.  Getting the head to actually fit that target
needed three fixes, each found by ``tools/syndrome_fit.py`` rather than by reading the code:

1. **same gauge on both sides.**  ``checks`` is LayerNorm-ed by the Tanner (per-element scale
   ~1) while ``parity_ref`` was a bare ``Linear`` (scale ~2).  The measured offset between
   them was 2.2x the effect the corruption has on ``checks`` (0.149 relative), so the head
   was reading gauge noise.  ``ref_norm`` puts both sides on the same scale.
2. **explicit base term.**  ``RMS(agg - ref)`` is added to the learned projection instead of
   relying on a 3-layer MLP to discover the scale of a small difference.
3. **biased start.**  The final sigmoid's bias is preset to ``logit(expected density)``
   instead of 0.  A bias of 0 starts at 0.5, whose soft-BCE is exactly ``ln 2 = 0.693`` -- and
   with a target of 0.2 the optimum is 2.7 logits away, so the head spends the whole run
   unlearning its own initialisation.

The metric that decides whether any of this worked is
``soft-BCE(S) < soft-BCE(constant = density.mean())``.  The pre-fix model scored 0.6932 vs a
constant's 0.6534 -- *worse than predicting the mean*, i.e. the term was actively harmful.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualSyndrome(nn.Module):
    """``s_j = D(phi({v_i : i in N(j)}), p_j)`` and its learned aggregation."""

    def __init__(self, dim: int = 768, check_dim: int = 128, num_parity: int = 16,
                 hidden: int = 128, discrepancy: str = "learned", code_dim: int = 256,
                 density_prior: float = 0.2):
        super().__init__()
        self.num_parity = num_parity
        self.discrepancy = discrepancy

        self.phi = nn.Sequential(
            nn.Linear(check_dim, check_dim), nn.GELU(), nn.Linear(check_dim, check_dim),
        )
        self.parity_ref = nn.Linear(code_dim, check_dim)     # codebook parity -> check space
        # ``checks`` and ``parity_ref`` are produced by different modules on different
        # inputs, so they land on different scales: measured ||agg|| = 0.46 ||ref||, i.e.
        # the *offset between the two gauges* was 2.2x the effect corruption has on ``checks``
        # (0.149 relative change).  Differencing them directly therefore reads gauge noise,
        # and the syndrome head provably collapsed to a constant -- it scored a soft-BCE of
        # 0.6932 (== ln 2, a constant 0.5) against a constant-mean predictor's 0.6534, i.e.
        # worse than useless.  Both sides are LayerNorm-ed so only the *direction* of the
        # discrepancy survives, which is what carries the corruption evidence.
        self.agg_norm = nn.LayerNorm(check_dim)
        self.ref_norm = nn.LayerNorm(check_dim)
        # The pre-normalisation residual: `H x` minus what the trusted identity predicts
        # for the same neighbourhood.  `checks` is LayerNorm-ed, so the absolute deviation
        # is normalised away and the corruption only moves it ~13%; this path keeps the
        # magnitude, which is the quantity the density target actually measures.
        self.residual_mlp = nn.Sequential(
            nn.Linear(check_dim, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )
        # gain on the observation-energy read-out.  Starts at 1.0 because that term is the
        # one measurement that provably carries the answer (corr = 1.000 with the density);
        # the network can zero it if it turns out to be useless, but it should not have to
        # discover it from a 0.13 signal-to-noise ratio.
        self.energy_gain = nn.Parameter(torch.tensor(1.0))
        self.d = nn.Sequential(
            nn.Linear(check_dim, hidden), nn.GELU(), nn.Linear(hidden, 1),
        ) if discrepancy == "learned" else None
        # learnable per-check scale on the discrepancy, so one check cannot dominate
        self.d_scale = nn.Parameter(torch.ones(num_parity))

        # SP (syndrome processing) x2, then ``S``
        self.sp1 = nn.Sequential(nn.Linear(num_parity, hidden), nn.GELU())
        self.sp2 = nn.Sequential(nn.Linear(hidden, num_parity), nn.Sigmoid())

        # Start at the density the supervision actually expects.  With a 0 bias the head
        # outputs 0.5 everywhere, soft-BCE sits at ln(2) = 0.693, and the observed symptom
        # is a loss that never moves (0.694 -> 0.706 over thousands of steps) because the
        # sigmoid's gradient vanishes long before the target's logit.
        prior = min(max(float(density_prior), 1e-3), 1 - 1e-3)
        with torch.no_grad():
            # sp2 is Sequential(Linear, Sigmoid): the bias to preset is sp2[-2]'s
            bias = self.sp2[-2].bias if isinstance(self.sp2[-2], nn.Linear) else None
            if bias is not None:
                bias.fill_(float(torch.logit(torch.tensor(prior))))

    def forward(self, checks: torch.Tensor, parity: torch.Tensor,
                residual: Optional[torch.Tensor] = None,
                obs_energy: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """``checks``: ``B x 16 x 128`` (already aggregated through ``H``);
        ``parity``: ``B x 16 x 256``; ``residual``: pre-norm ``H x - expected``, ``B x 16 x 128``;
        ``obs_energy``: per-check observation energy, ``B x 16``."""
        agg = self.agg_norm(self.phi(checks))                    # B x 16 x 128
        ref = self.ref_norm(self.parity_ref(parity))              # B x 16 x 128, same gauge

        # residual discrepancy on the normalised side: direction only
        delta = agg - ref                                          # B x 16 x 128
        mean_sq = delta.pow(2).mean(dim=-1)
        if self.d is not None:
            base = mean_sq.sqrt()                                  # RMS(agg - ref)
            s = base + self.d(delta).squeeze(-1)
        else:
            s = mean_sq                                             # plain L2, as before

        # The magnitude path.  Erasure zeroes a token, so ``mean_i ||x_i||^2`` over a check's
        # neighbourhood IS the density (measured corr = 1.000).  ``checks`` cannot carry it:
        # it is LayerNorm-ed, which fixes the per-element scale, and the corruption only
        # moves it ~13% relative -- against that noise the head provably collapsed to a
        # constant (soft-BCE 0.6932 == ln 2, worse than a constant-mean predictor's 0.6534).
        # Feeding the raw energy bypasses the normalisation that was erasing the evidence.
        if obs_energy is not None:
            s = s + self.energy_gain * obs_energy.float()
        if residual is not None:
            res = residual.float()
            res_energy = res.pow(2).mean(dim=-1).sqrt()             # B x 16, raw scale
            res_energy = res_energy / res_energy.mean(dim=-1, keepdim=True).clamp(min=1e-6)
            s = s + self.residual_mlp(res).squeeze(-1) + res_energy

        s_raw = s * self.d_scale                                  # B x 16

        y = self.sp1(s_raw)
        syndrome = self.sp2(y)                                   # B x 1 x 16

        return {"syndrome": syndrome.unsqueeze(1), "syndrome_raw": s_raw,
                "check_agg": agg, "check_ref": ref, "check_delta": delta,
                "check_residual": residual}


class ErrorLocator(nn.Module):
    """Turn a syndrome pattern into a per-variable corruption probability."""

    def __init__(self, num_parity: int = 16, num_variables: int = 256,
                 num_graph_nodes: int = 128, hidden: int = 128):
        super().__init__()
        self.num_variables = num_variables
        self.num_graph_nodes = num_graph_nodes
        # syndrome -> graph-node logits, then scatter to variables through the graph
        self.syndrome_to_node = nn.Sequential(
            nn.Linear(num_parity, hidden), nn.GELU(), nn.Linear(hidden, num_graph_nodes),
        )
        self.node_to_variable = nn.Linear(num_graph_nodes, num_variables)

    def forward(self, syndrome: torch.Tensor, identity_map: torch.Tensor,
                node_index: torch.Tensor, num_variables: int,
                H: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """``locator`` is ``B x 128 x 256`` (per graph node -> per variable evidence),
        ``locator_scattered`` is ``B x 256`` -- produced by pushing the syndrome back
        through the **shared** ``H``, so "checks 2, 7 and 9 fired, and all of them watch
        v17" is literally what the computation does.  ``severity`` is ``B x 128``.

        The ``H^T`` vote uses the **frame-centred** syndrome.  ``syndrome`` is trained to
        equal a corruption *density* whose mean is the corruption ratio, so its absolute
        level is already pinned by ``L_detect``'s soft-BCE.  Feeding that same raw tensor
        into the localization BCE creates a direct tug-of-war over one variable: with a
        constant ``S = 0.5`` the scattered vote is a constant 0.5, which already scores a
        respectable BCE against a 0.4-positive mask, and measured over training it pinned
        ``S`` at 0.70 while the reliability term fell 1.36 -> 0.17.  Removing the frame
        mean leaves the localization loss asking only "which checks are *worse than
        average*", which is scale-free and cannot fight the calibration.
        """
        s = syndrome.flatten(1)                                   # B x M
        node_logits = self.syndrome_to_node(s)                    # B x 128
        variable_prior = self.node_to_variable(node_logits)       # B x 256
        # every graph node spreads its evidence over the variable nodes
        locator = torch.sigmoid(node_logits.unsqueeze(-1) * variable_prior.unsqueeze(1))
        # only nodes with a weak identity match are corruption suspects
        suspect = (1.0 - identity_map).unsqueeze(-1)              # B x 128 x 1
        locator = locator * suspect
        severity = (1.0 - identity_map) * torch.sigmoid(node_logits)   # B x 128

        if H is not None:
            # H^T: each failing check votes for exactly the variables it watches.
            # Centred per frame (and rescaled to a usable range) so the vote measures
            # relative damage, which is what a variable-level label can supervise.
            centred = s - s.mean(dim=1, keepdim=True)
            scale = centred.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
            vote = 0.5 + 0.5 * (centred / scale)                   # ~[0, 1], 0.5 = average
            scattered = (vote @ H).clamp(0.0, 1.0)                # B x 256
        else:
            node_score = locator.max(dim=2).values                # B x 128
            scattered = torch.zeros(locator.shape[0], num_variables,
                                    device=locator.device, dtype=locator.dtype)
            scattered.scatter_(1, node_index, node_score)
        return {"locator": locator, "locator_scattered": scattered,
                "severity": severity, "node_logits": node_logits}


class ReliabilityAwareGating(nn.Module):
    """Error severity + **per-modality** reliability -> per-node correction strength.

    Two gates are produced because RGB and TIR are repaired by the same Tanner graph
    but have independent reliability estimates; sharing one gate would let a corrupted
    modality be "corrected" as if it were healthy.
    """

    def __init__(self, num_graph_nodes: int = 128, hidden: int = 64):
        super().__init__()
        self.num_graph_nodes = num_graph_nodes
        self.mlp = nn.Sequential(
            nn.Linear(num_graph_nodes * 3, hidden), nn.GELU(),
            nn.Linear(hidden, num_graph_nodes * 2), nn.Sigmoid(),
        )

    def forward(self, severity: torch.Tensor, reliability_rgb: torch.Tensor,
                reliability_tir: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``severity``/``reliability_*``: ``B x 128`` -> ``(gate_rgb, gate_tir)``."""
        x = torch.cat([severity, reliability_rgb, reliability_tir], dim=-1)
        out = self.mlp(x)
        return out[..., :self.num_graph_nodes], out[..., self.num_graph_nodes:]
