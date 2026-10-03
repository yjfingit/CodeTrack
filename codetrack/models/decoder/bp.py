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
                 mode: str = "bp", mlp_hidden: int = 0,
                 locality_window: int = 0, free_edge_frac: float = 0.25,
                 locality_wrap: bool = True,
                 generator: Optional[torch.Generator] = None):
        super().__init__()
        self.dim = dim
        self.iterations = iterations
        self.num_parity = num_parity
        self.num_variables = num_variables
        self.mode = mode

        # ---- parity-check matrix H (M x N) with a fixed sparse support -------------
        support = torch.zeros(num_parity, num_variables)
        # Locality prior.  With a uniformly random support, a spatially contiguous burst is
        # still just "an arbitrary fixed-size set" to every check, so the burst's spatial
        # continuity is washed out on the way into check-space -- measured, the per-check
        # density spread under burst (0.049) was no larger than under independent erasure
        # (0.058).  The Tanner graph was therefore sensing combinatorial adjacency, not visual
        # space.  Measured with the bypass off, burst is in fact the ONE corruption where the
        # syndrome is informative (pearson 0.793 vs 0.007), so some of that alignment does
        # matter; this makes it a *prior* rather than a hard constraint.
        #
        # Each check draws its edges from a local candidate window (row * cand -> next
        # cand * grid cells on the 2-D search grid) and still LEARNS which k of those to
        # keep via softplus(H).  A small share of edges stays unrestricted so the model can
        # reach across the frame when locality is the wrong bias.  The claim stays "learned
        # sparse H under a geometric prior", not "hand-designed neighbourhood".
        grid = int(round(num_variables ** 0.5))
        spatial = (grid * grid == num_variables) and (grid >= 2)
        if spatial and locality_window > 0:
            cand = min(int(locality_window), num_variables)
            # window start depends on the check, so different checks watch different places
            span = max(num_variables - cand, 1)
            cand_idx = torch.arange(num_variables).view(1, -1)      # 1 x N
            offset = (torch.rand(num_parity, generator=generator) * span).long().view(-1, 1)
            dist = (cand_idx - offset).abs()                        # M x N
            # circular distance on the grid, so the window wraps instead of biasing the
            # right/bottom edge
            if locality_wrap:
                dist = torch.minimum(dist, num_variables - dist)
            allowed = dist < cand
            for row in range(num_parity):
                pool = torch.nonzero(allowed[row], as_tuple=False).flatten()
                if pool.numel() == 0:
                    pool = torch.randperm(num_variables, generator=generator)
                picked = pool[torch.randperm(pool.numel(), generator=generator)
                              [:min(links_per_check, pool.numel())]]
                support[row, picked] = 1.0
            # the unrestricted share: a few random edges per check, so a check can still
            # reach outside its window when the geometry is misleading
            n_free = max(1, int(round(free_edge_frac * links_per_check)))
            for row in range(num_parity):
                free = torch.nonzero(support[row] == 0).flatten()
                if free.numel() == 0:
                    continue
                picked = free[torch.randperm(free.numel(), generator=generator)[:n_free]]
                support[row, picked] = 1.0
        else:
            for row in range(num_parity):
                picked = torch.randperm(num_variables, generator=generator)[:links_per_check]
                support[row, picked] = 1.0

        # Guarantee a minimum column degree.  A variable watched by no check can never
        # receive a correction message, so its "repair" could only come from the local
        # update MLP -- which would silently turn the BP path into decoration.
        # Attach to *free* rows, and loop, so the guarantee holds even when the random
        # pattern is sparse relative to num_parity * min_column_degree.
        col_degree = support.sum(dim=0)
        for col in torch.nonzero(col_degree < min_column_degree, as_tuple=False).flatten():
            while float(support[:, col].sum()) < min_column_degree:
                free = torch.nonzero(support[:, col] == 0).flatten()
                if free.numel() == 0:
                    break
                pick = int(torch.randint(free.numel(), (1,), generator=generator).item())
                support[int(free[pick]), col] = 1.0

        self.register_buffer("H_support", support)
        self.H = nn.Parameter(support.clone() / links_per_check)

        # ---- message functions ---------------------------------------------------
        # In "mlp" mode none of these exist, otherwise they would be dead parameters and
        # the baseline would not be parameter-matched.
        if mode == "bp":
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
        else:
            self.v_msg = self.parity_proj = self.c_msg = self.update = None
        self.out_norm = nn.LayerNorm(dim)

        # ---- parameter-matched baseline -------------------------------------
        # "mlp" removes message passing entirely and replaces each BP round with a
        # residual MLP of matched width.  If this matches the BP decoder, the whole
        # Tanner-graph story is decoration.
        # One BP round costs roughly 2 * dim * (3 * dim) parameters; a two-layer MLP
        # of the same input width costs 3 * hidden * dim, hence hidden ~ 2.7 * dim.
        if mlp_hidden <= 0:
            mlp_hidden = max(8, int(round(dim * 2048 / 768)))
        self.mlp_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim + 1, mlp_hidden), nn.GELU(),   # input is [v, r]
                nn.Linear(mlp_hidden, dim),
            ) for _ in range(iterations)
        ]) if mode == "mlp" else None

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
        v = torch.cat([variables_rgb, variables_tir], dim=0)       # 2B x 256 x 768
        r = torch.cat([reliability_rgb, reliability_tir], dim=0).unsqueeze(-1)
        total_delta = torch.zeros_like(v)

        if self.mode == "mlp":
            # no parity-check matrix, no messages: a matched residual denoiser
            for block in self.mlp_blocks:
                delta = block(torch.cat([v, r], dim=-1))
                v = v + (1.0 - r) * delta
                total_delta = total_delta + (1.0 - r) * delta
            v = self.out_norm(v)
            return {
                "corrected_rgb": v[:b], "corrected_tir": v[b:],
                "residual_rgb": total_delta[:b], "residual_tir": total_delta[b:],
            }

        # ---- belief propagation ---------------------------------------------------
        h = self._h()                                              # M x N
        parity_c = parity.repeat(2, 1, 1) if parity.shape[0] == b else parity
        s = syndrome.repeat(2, 1, 1) if syndrome.shape[0] == b else syndrome
        parity_ctx = self.parity_proj(parity_c)                    # 2B x M x dim
        s_col = s.flatten(1).unsqueeze(-1)                         # 2B x M x 1

        m_cv = torch.zeros_like(v)

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

        # norm-only control: the decoder's output LayerNorm applied to the *uncorrected*
        # input.  It isolates the pure scale change from the message updates, so a negative
        # recovery_gain can be attributed (or not) to the Tanner messages.
        norm_only = self.out_norm(torch.cat([variables_rgb, variables_tir], dim=0))
        v = self.out_norm(v)
        return {
            "corrected_rgb": v[:b],
            "corrected_tir": v[b:],
            "residual_rgb": total_delta[:b],
            "residual_tir": total_delta[b:],
            "norm_only_rgb": norm_only[:b],
            "norm_only_tir": norm_only[b:],
        }
