"""Neural belief-propagation decoder -- architecture figure block 6.

    l-th iteration
        Variable -> Check:  m_{i->j}^{(l)} = f_v( v_i^{(l)}, r_i, m_{k->i}^{(l-1)} )
        Check -> Variable:  m_{j->i}^{(l)} = f_c( s_j, p_j, {m_{k->j}} )
        Update:             v_i^{(l+1)} = v_i^{(l)} + (1 - r_i) * dv_i^{(l)}

    Corrected Variable Nodes                       B x 256 x 768
    Residual                                       B x 256 x 768

The parity-check matrix ``H`` (``16 x 256``, sparse support) is the Tanner graph
itself: ``m_vc = H v`` aggregates variables into checks, ``m_cv = H^T f_c(...)`` sends
the correction back.  ``(1 - r_i)`` is the reliability gate: a confident token barely
moves, an unreliable one absorbs the correction.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralBPDecoder(nn.Module):
    """Iterative message passing over the parity-check matrix."""

    def __init__(self, dim: int = 768, code_dim: int = 256, num_parity: int = 16,
                 num_variables: int = 256, iterations: int = 2,
                 links_per_check: int = 32, min_column_degree: int = 2,
                 generator: Optional[torch.Generator] = None):
        super().__init__()
        self.dim = dim
        self.iterations = iterations
        self.num_parity = num_parity
        self.num_variables = num_variables

        # ---- parity-check matrix H (16 x 256) with a fixed sparse support ----------
        support = torch.zeros(num_parity, num_variables)
        for row in range(num_parity):
            picked = torch.randperm(num_variables, generator=generator)[:links_per_check]
            support[row, picked] = 1.0

        # Guarantee a minimum column degree.  A variable watched by no check can never
        # receive a correction message, so its "repair" could only come from the local
        # update MLP -- which would silently turn the BP path into decoration.
        col_degree = support.sum(dim=0)
        for col in torch.nonzero(col_degree < min_column_degree, as_tuple=False).flatten():
            need = int(min_column_degree - col_degree[col])
            rows = torch.randperm(num_parity, generator=generator)[:need]
            support[rows, col] = 1.0

        self.register_buffer("H_support", support)
        self.H = nn.Parameter(support.clone() / links_per_check)

        # ---- message functions ---------------------------------------------------
        self.v_msg = nn.Sequential(
            nn.Linear(dim + 1, dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.parity_proj = nn.Linear(code_dim, dim)
        self.c_msg = nn.Sequential(
            nn.Linear(dim * 2 + 1, dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.update = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

    # ------------------------------------------------------------------ helpers
    def _h(self) -> torch.Tensor:
        # non-negative, row-normalised weights: softplus keeps A_ij in R+ as documented,
        # whereas a raw Parameter can drift negative and explode when its row sum -> 0
        h = F.softplus(self.H) * self.H_support
        return h / h.sum(dim=1, keepdim=True).clamp(min=1e-6)

    def connectivity(self) -> torch.Tensor:
        return self.H_support

    def matrix(self) -> torch.Tensor:
        """The normalised parity-check matrix, ``M x N`` (16 x 256).

        Exposed so that the syndrome target, the locator and any analysis code all use
        the *same* incidence the decoder propagates messages over.
        """
        return self._h()

    # ------------------------------------------------------------------ forward
    def forward(self, variables_rgb: torch.Tensor, variables_tir: torch.Tensor,
                parity: torch.Tensor, syndrome: torch.Tensor,
                reliability_rgb: torch.Tensor, reliability_tir: torch.Tensor,
                gate_rgb: Optional[torch.Tensor] = None,
                gate_tir: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Dual-modality message passing over one **shared** Tanner graph.

        ``variables_rgb`` / ``variables_tir``  ``B x 256 x 768``
        ``parity``      ``B x 16 x 256``
        ``syndrome``    ``B x 1 x 16``
        ``reliability_*``  ``B x 256``
        ``gate``        ``B x 128`` (optional, from the severity path)

        Returns ``corrected_rgb``, ``corrected_tir``, ``residual_rgb``, ``residual_tir``,
        each ``B x 256 x 768`` -- matching "Corrected zg / Corrected xg / Residual" in
        the architecture figure.
        """
        b = variables_rgb.shape[0]
        h = self._h()                                              # 16 x 256

        v = torch.cat([variables_rgb, variables_tir], dim=0)       # 2B x 256 x 768
        r = torch.cat([reliability_rgb, reliability_tir], dim=0).unsqueeze(-1)
        parity_c = parity.repeat(2, 1, 1) if parity.shape[0] == b else parity
        s = syndrome.repeat(2, 1, 1) if syndrome.shape[0] == b else syndrome
        parity_ctx = self.parity_proj(parity_c)                    # 2B x 16 x 768
        s_col = s.flatten(1).unsqueeze(-1)                         # 2B x 16 x 1

        m_cv = torch.zeros_like(v)
        total_delta = torch.zeros_like(v)

        for _ in range(self.iterations):
            # ---- Variable -> Check ------------------------------------------------
            m_vc_full = self.v_msg(torch.cat([v, r], dim=-1))      # 2B x 256 x 768
            m_vc = torch.einsum("cv,bvd->bcd", h, m_vc_full)       # 2B x 16 x 768

            # ---- Check -> Variable ------------------------------------------------
            c_in = torch.cat([m_vc, parity_ctx, s_col.expand(-1, -1, 1)], dim=-1)
            c_out = self.c_msg(c_in)                               # 2B x 16 x 768
            m_cv = torch.einsum("cv,bcd->bvd", h, c_out)           # 2B x 256 x 768

            # ---- Update: v <- v + (1 - r) * gate * delta ---------------------------
            delta = self.update(torch.cat([v, m_cv], dim=-1))      # 2B x 256 x 768
            if gate_rgb is not None or gate_tir is not None:
                g_rgb = gate_rgb if gate_rgb is not None else torch.ones_like(reliability_rgb)
                g_tir = gate_tir if gate_tir is not None else torch.ones_like(reliability_tir)
                g = torch.cat([g_rgb, g_tir], dim=0) if g_rgb.shape[0] == b else g_rgb
                g = F.interpolate(g.unsqueeze(1), size=v.shape[1], mode="linear",
                                  align_corners=False).transpose(1, 2)
                delta = delta * g
            v = v + (1.0 - r) * delta
            total_delta = total_delta + (1.0 - r) * delta

        v = self.out_norm(v)
        return {
            "corrected_rgb": v[:b],
            "corrected_tir": v[b:],
            "residual_rgb": total_delta[:b],
            "residual_tir": total_delta[b:],
        }
