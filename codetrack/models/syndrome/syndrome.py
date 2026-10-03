"""Visual Syndrome, Error Locator and severity gating -- architecture figure block 5.

    SP -> SP -> Syndrome S                B x 1 x 16
    Error Locator                         B x 128 x 256
    Error Severity (Reliability-Aware Gating)   B x 128

The syndrome is the conceptual core: for a healthy code it is ~0, for a corrupted
token the checks that watch it light up.  Two capabilities follow from it:

* **detection** -- ``S -> P(corruption)``;
* **localization** -- because ``A`` is sparse, the *pattern* of failing checks votes
  for a specific variable node.  Plain uncertainty fusion cannot do this.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualSyndrome(nn.Module):
    """``s_j = D(phi({v_i : i in N(j)}), p_j)`` and its learned aggregation."""

    def __init__(self, dim: int = 768, check_dim: int = 128, num_parity: int = 16,
                 hidden: int = 128, discrepancy: str = "learned", code_dim: int = 256):
        super().__init__()
        self.num_parity = num_parity
        self.discrepancy = discrepancy

        self.phi = nn.Sequential(
            nn.Linear(check_dim, check_dim), nn.GELU(), nn.Linear(check_dim, check_dim),
        )
        self.parity_ref = nn.Linear(code_dim, check_dim)     # codebook parity -> check space
        self.d = nn.Sequential(
            nn.Linear(check_dim, hidden), nn.GELU(), nn.Linear(hidden, 1),
        ) if discrepancy == "learned" else None

        # SP (syndrome processing) x2, then ``S``
        self.sp1 = nn.Sequential(nn.Linear(num_parity, hidden), nn.GELU())
        self.sp2 = nn.Sequential(nn.Linear(hidden, num_parity), nn.Sigmoid())

    def forward(self, checks: torch.Tensor, parity: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        """``checks``: ``B x 16 x 128`` (already aggregated through ``H``);
        ``parity``: ``B x 16 x 256``."""
        agg = self.phi(checks)                                    # B x 16 x 128
        ref = self.parity_ref(parity)                             # B x 16 x 128

        if self.d is not None:
            s = self.d(agg - ref).squeeze(-1)                     # learned discrepancy
        else:
            s = (agg - ref).pow(2).mean(dim=-1)                   # L2 discrepancy

        s_raw = s                                                # B x 16
        y = self.sp1(s_raw)
        syndrome = self.sp2(y)                                   # B x 1 x 16
        return {"syndrome": syndrome.unsqueeze(1), "syndrome_raw": s_raw,
                "check_agg": agg, "check_ref": ref}


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
        v17" is literally what the computation does.  ``severity`` is ``B x 128``."""
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
            # H^T: each failing check votes for exactly the variables it watches
            scattered = (s @ H).clamp(0.0, 1.0)                   # B x 256
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
