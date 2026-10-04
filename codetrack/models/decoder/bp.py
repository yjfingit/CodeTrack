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
                 output_mode: str = "post_norm", residual_clip: float = 0.0,
                 gate_always_one: bool = False,
                 locality_window: int = 0, free_edge_frac: float = 0.25,
                 locality_wrap: bool = True, weight_init: str = "learned",
                 balance_degrees: bool = False,
                 generator: Optional[torch.Generator] = None):
        super().__init__()
        self.dim = dim
        self.iterations = iterations
        self.num_parity = num_parity
        self.num_variables = num_variables
        self.mode = mode
        self.balance_degrees = bool(balance_degrees)

        # ---- parity-check matrix H (M x N): a FIXED sparse support, LEARNED weights ----
        #
        # Naming matters here and was wrong before.  ``H_support`` is registered as a buffer
        # and never updated: ``softplus(H) * H_support`` cannot create an edge, and nothing
        # runs a top-k selection during training.  The honest description is therefore
        # **"learned edge weights on a geometrically-constructed fixed sparse support"**,
        # not "learned connectivity".
        #
        # Locality prior. Each check draws candidate edges from a search-grid window. The
        # optional degree-matching path splits the requested budget between local and free
        # edges; the default legacy sampler is retained when the flag is disabled.
        support = torch.zeros(num_parity, num_variables)
        local_allowed = torch.ones(num_parity, num_variables, dtype=torch.bool)
        grid = int(round(num_variables ** 0.5))
        spatial = (grid * grid == num_variables) and (grid >= 2)
        if spatial and locality_window > 0:
            # 2-D grid distance.  The previous code used ``|index - offset|`` on the *flattened*
            # index, which is not a spatial distance at all: a window of 16 there is up to 31
            # consecutive indices, i.e. it straddles row ends, and "wrap" by
            # ``num_variables - dist`` is meaningless on a flattened layout.
            yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
            coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1).float()   # N x 2
            half = max(1, int(locality_window) // 2)

            for row in range(num_parity):
                # a different centre per check, so the checks do not all watch one place
                cy = int(torch.randint(0, grid, (1,), generator=generator))
                cx = int(torch.randint(0, grid, (1,), generator=generator))
                dy = (coords[:, 0] - cy).abs()
                dx = (coords[:, 1] - cx).abs()
                if locality_wrap:
                    dy = torch.minimum(dy, grid - dy)
                    dx = torch.minimum(dx, grid - dx)
                inside = torch.nonzero((dy <= half) & (dx <= half), as_tuple=False).flatten()
                local_allowed[row] = False
                local_allowed[row, inside] = True
                n_free = int(round(free_edge_frac * links_per_check))
                n_local = links_per_check - n_free if self.balance_degrees else links_per_check
                pool = inside if inside.numel() >= n_local else torch.arange(
                    num_variables)
                picked = pool[torch.randperm(pool.numel(), generator=generator)[:n_local]]
                support[row, picked] = 1.0

            # In legacy mode, add the unrestricted share after the requested local edges. In
            # degree-matched mode, the local selection above already reserved this share.
            n_free = int(round(free_edge_frac * links_per_check))
            for row in range(num_parity):
                free = torch.nonzero(support[row] == 0).flatten()
                if free.numel() == 0 or n_free == 0:
                    continue
                picked = free[torch.randperm(free.numel(), generator=generator)[:n_free]]
                support[row, picked] = 1.0
        else:
            for row in range(num_parity):
                picked = torch.randperm(num_variables, generator=generator)[:links_per_check]
                support[row, picked] = 1.0

        # Minimum column degree. A variable watched by no check cannot receive a correction
        # message. The legacy sampler adds repair edges and may change realized row degrees;
        # degree-matched runs swap edges instead and require a feasible total edge budget.
        col_degree = support.sum(dim=0)
        if self.balance_degrees:
            if links_per_check * num_parity < min_column_degree * num_variables:
                raise ValueError(
                    "balanced support cannot meet min_column_degree: increase "
                    "links_per_check or lower min_column_degree"
                )
            # Add a missing incidence by swapping it for an incidence on a column whose
            # degree exceeds the minimum.  This keeps each row's degree fixed; where
            # possible it also preserves the local/free-edge category of that row.
            for col in torch.nonzero(col_degree < min_column_degree,
                                     as_tuple=False).flatten().tolist():
                while int(col_degree[col]) < min_column_degree:
                    rows = torch.nonzero(support[:, col] == 0,
                                         as_tuple=False).flatten()
                    rows = rows[torch.randperm(rows.numel(), generator=generator)]
                    swapped = False
                    for row_t in rows:
                        row = int(row_t)
                        donors = torch.nonzero(
                            support[row].bool() & (col_degree > min_column_degree),
                            as_tuple=False).flatten()
                        if donors.numel() == 0:
                            continue
                        same_kind = local_allowed[row, donors] == local_allowed[row, col]
                        preferred = donors[same_kind]
                        choices = preferred if preferred.numel() else donors
                        pick = int(torch.randint(choices.numel(), (1,),
                                                 generator=generator).item())
                        donor = int(choices[pick])
                        support[row, donor] = 0.0
                        support[row, col] = 1.0
                        col_degree[donor] -= 1.0
                        col_degree[col] += 1.0
                        swapped = True
                        break
                    if not swapped:
                        raise RuntimeError("could not balance parity-check column degrees")
        else:
            for col in torch.nonzero(col_degree < min_column_degree,
                                     as_tuple=False).flatten():
                while float(support[:, col].sum()) < min_column_degree:
                    free = torch.nonzero(support[:, col] == 0).flatten()
                    if free.numel() == 0:
                        break
                    pick = int(torch.randint(free.numel(), (1,), generator=generator).item())
                    support[int(free[pick]), col] = 1.0

        self.register_buffer("H_support", support)
        # ``weight_init="uniform"`` freezes the edge weights at 1/degree, i.e. plain
        # averaging over the neighbourhood.  It exists so the paper can separate the two
        # contributions the review asked about: a *fixed local support with uniform weights*
        # is pure geometry, the same support with learned weights adds learning on top.
        self.learn_weights = str(weight_init) != "uniform"
        self.H = nn.Parameter(support.clone() / links_per_check,
                              requires_grad=self.learn_weights)

        # ---- message functions ---------------------------------------------------
        # In "mlp"/"spatial" mode none of these exist, otherwise they would be dead parameters
        # and the baseline would not be parameter-matched.
        if mode not in ("bp", "mlp", "spatial", "off"):
            raise ValueError(f"unknown decoder mode {mode!r}")
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
        # mode "off" has no message functions and no output norm: it is a pass-through, and
        # leaving a learnable LayerNorm in place would let "no decoder" still rescale tokens.
        self.out_norm = nn.LayerNorm(dim) if mode != "off" else None

        # ---- output parameterisation -------------------------------------------
        # "post_norm" (default) is the shipped behaviour: the update happens in the raw token
        # space and the *output* is normalised, while ``L_correct`` compares that output
        # against pre-norm clean features.  The probe in ``tools/step_size_probe.py`` shows
        # what that costs: the update overshoots the ideal step by 2-10x and the LayerNorm
        # absorbs the scale, so the optimiser can "fix" the loss by shrinking the update.
        #
        # "identity_residual" moves the normalisation *inside* the update branch and leaves the
        # output un-normalised:
        #     u  = LN_branch(v)                              (pre-norm on the branch input)
        #     dv = W_o * f_theta([u, messages]) + b_o        (W_o = 0, b_o = 0 at init)
        #     v' = v + (1 - r) * gate * C_B(dv)
        # so the initial output is the input (a true identity path), the reported residual is
        # the residual that is actually applied, and a bounded step cannot silently overshoot.
        self.output_mode = str(output_mode)
        if self.output_mode not in ("post_norm", "identity_residual"):
            raise ValueError(f"unknown decoder output_mode {output_mode!r}")
        self.residual_clip = float(residual_clip)
        self.gate_always_one = bool(gate_always_one)
        # Counts steps whose update came out non-finite and were replaced by zero.  A plain int,
        # not a buffer: a buffer would add a key to the state dict and make every existing
        # checkpoint unloadable.  The identity-residual arms train an un-normalised residual sum
        # under fp16 autocast, where one overflowing MLP output can turn into inf -> NaN and then
        # abort the whole run with a device-side assert (the loss then sees a NaN probability and
        # BCELoss asserts); this makes that an event that is counted and logged instead of fatal.
        self.nonfinite_steps = 0
        if mode != "off" and self.output_mode == "identity_residual":
            self.branch_norm = nn.LayerNorm(dim)
            self.residual_out = nn.Linear(dim, dim)
            nn.init.zeros_(self.residual_out.weight)
            nn.init.zeros_(self.residual_out.bias)
        else:
            self.branch_norm = None
            self.residual_out = None

        # ---- parameter-matched baselines -------------------------------------
        # "mlp" removes message passing entirely and replaces each BP round with a residual MLP
        # of matched width; "spatial" additionally gives that MLP a *local* receptive field
        # (depthwise 3x3 over the token grid) so the comparison separates "Tanner structure" from
        # "ordinary spatial context".  Both match the BP message-function parameter count.
        if mlp_hidden <= 0:
            def linear_params(in_features: int, out_features: int) -> int:
                return in_features * out_features + out_features

            bp_message_params = (
                linear_params(dim + 1, dim) + linear_params(dim, dim)
                + linear_params(code_dim, dim)
                + linear_params(2 * dim + 1, dim) + linear_params(dim, dim)
                + linear_params(2 * dim, dim) + linear_params(dim, dim)
            )
            n_iterations = max(1, iterations)
            mlp_params_per_hidden = n_iterations * (2 * dim + 2)
            mlp_fixed_params = n_iterations * dim
            spatial_params = (dim * 9 + dim) * n_iterations if mode == "spatial" else 0
            mlp_hidden = max(
                8,
                int(round((bp_message_params - mlp_fixed_params - spatial_params) /
                          mlp_params_per_hidden)),
            )
        self.mlp_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim + 1, mlp_hidden), nn.GELU(),   # input is [v, r]
                nn.Linear(mlp_hidden, dim),
            ) for _ in range(iterations)
        ]) if mode in ("mlp", "spatial") else None
        self.mlp_hidden = mlp_hidden

        # "spatial": a depthwise 3x3 convolution over the (grid x grid) token layout, i.e. the
        # cheapest possible way to give the mixer the local neighbourhood the Tanner checks are
        # built from -- without any parity, syndrome or locator quantity.
        self.grid = int(round(num_variables ** 0.5))
        if mode == "spatial":
            if self.grid * self.grid != num_variables:
                raise ValueError(
                    f"decoder_mode='spatial' needs a square token grid, got "
                    f"num_variables={num_variables}")
            self.spatial_conv = nn.ModuleList([
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
                for _ in range(iterations)
            ])
        else:
            self.spatial_conv = None

    # ------------------------------------------------------------------ helpers
    def _h(self) -> torch.Tensor:
        # non-negative, row-normalised weights: softplus keeps A_ij in R+ as documented,
        # whereas a raw Parameter can drift negative and explode when its row sum -> 0
        h = (F.softplus(self.H) if self.learn_weights
             else torch.full_like(self.H, 1.0)) * self.H_support
        return h / h.sum(dim=1, keepdim=True).clamp(min=1e-6)

    def connectivity(self) -> torch.Tensor:
        return self.H_support

    def matrix(self) -> torch.Tensor:
        """The normalised parity-check matrix, ``M x N`` (16 x 256).

        Exposed so that the syndrome target, the locator and any analysis code all use
        the *same* incidence the decoder propagates messages over.
        """
        return self._h()

    @staticmethod
    def _expand_gate(gate: Optional[torch.Tensor], node_index: Optional[torch.Tensor],
                     batch: int, num_variables: int, reference: torch.Tensor
                     ) -> torch.Tensor:
        """Map selected-node gates back to the variable-token grid.

        The graph nodes are sorted by score, so interpolating their values would assign
        gates to unrelated spatial positions.  Variables omitted from the selected graph
        receive the neutral multiplier 1; their reliability gate ``(1-r)`` still applies.
        """
        if gate is None:
            return reference.new_ones((batch, num_variables))
        gate = gate.to(device=reference.device, dtype=reference.dtype)
        if gate.shape == (batch, num_variables):
            return gate
        if node_index is None or gate.shape != node_index.shape:
            raise ValueError(
                "gate must cover every variable or match node_index for scatter mapping"
            )
        index = node_index.to(device=reference.device, dtype=torch.long)
        if bool(((index < 0) | (index >= num_variables)).any()):
            raise ValueError("node_index contains an out-of-range variable index")
        expanded = reference.new_ones((batch, num_variables))
        return expanded.scatter(1, index, gate)

    # ------------------------------------------------------------------ forward
    def forward(self, variables_rgb: torch.Tensor, variables_tir: torch.Tensor,
                parity: torch.Tensor, syndrome: torch.Tensor,
                reliability_rgb: torch.Tensor, reliability_tir: torch.Tensor,
                gate_rgb: Optional[torch.Tensor] = None,
                gate_tir: Optional[torch.Tensor] = None,
                node_index: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Dual-modality message passing over one **shared** Tanner graph.

        ``variables_rgb`` / ``variables_tir``  ``B x 256 x 768``
        ``parity``      ``B x 16 x 256``
        ``syndrome``    ``B x 1 x 16``
        ``reliability_*``  ``B x 256``
        ``gate``        ``B x K`` (optional, from the severity path); ``node_index`` maps
                        those K selected graph nodes back to the N-token variable grid.

        Returns ``corrected_rgb``, ``corrected_tir``, ``residual_rgb``, ``residual_tir``,
        each ``B x 256 x 768`` -- matching "Corrected zg / Corrected xg / Residual" in
        the architecture figure.
        """
        # "off" = the correction branch is removed from the graph entirely.  Zeroing the
        # *loss weights* is not the same thing: the decoder would still run and still rewrite
        # the tokens, so a "no decoder" arm trained that way measures a normally-trained
        # decoder that is merely unpenalised.  tools/diagnostics.py uses the same definition,
        # so training and inference mean the same thing by "no decoder".
        if self.mode == "off":
            zeros = torch.zeros_like(variables_rgb)
            return {"corrected_rgb": variables_rgb, "corrected_tir": variables_tir,
                    "residual_rgb": zeros, "residual_tir": zeros,
                    "norm_only_rgb": variables_rgb, "norm_only_tir": variables_tir,
                    "pre_norm_rgb": variables_rgb, "pre_norm_tir": variables_tir}

        b = variables_rgb.shape[0]
        v = torch.cat([variables_rgb, variables_tir], dim=0)       # 2B x 256 x 768
        r = torch.cat([reliability_rgb, reliability_tir], dim=0).unsqueeze(-1)
        gate_r = self._expand_gate(gate_rgb, node_index, b, v.shape[1], variables_rgb)
        gate_t = self._expand_gate(gate_tir, node_index, b, v.shape[1], variables_tir)
        if self.gate_always_one:
            # The learned severity gate is an *extra* attenuator on top of (1 - r);
            # tools/step_size_probe.py measures that permuting its values within
            # (reliability, ||d||) strata changes nothing.  This switch removes it so the
            # four-arm grid can test "is the second gate worth anything at all".
            gate_r = torch.ones_like(gate_r)
            gate_t = torch.ones_like(gate_t)
        gate_full = torch.cat([gate_r, gate_t], dim=0).unsqueeze(-1)
        total_delta = torch.zeros_like(v)
        identity_output = self.output_mode == "identity_residual"
        branch_input = self.branch_norm(v) if identity_output else v

        if self.mode in ("mlp", "spatial"):
            # no parity-check matrix, no messages, no syndrome: a matched residual denoiser.
            # "spatial" additionally mixes each token with its 3x3 neighbourhood on the grid.
            for round_index, block in enumerate(self.mlp_blocks):
                features = branch_input
                if self.mode == "spatial":
                    conv = self.spatial_conv[round_index]
                    grid = self.grid
                    maps = features.transpose(1, 2).reshape(-1, self.dim, grid, grid)
                    mixed = conv(maps).reshape(-1, self.dim, grid * grid).transpose(1, 2)
                    features = features + mixed
                delta = self._step(block(torch.cat([features, r], dim=-1)))
                delta = (1.0 - r) * gate_full * delta
                v = v + delta
                total_delta = total_delta + delta
                if identity_output:
                    # pre-norm branch: the next step reads the normalised *current* state
                    branch_input = self.branch_norm(v)
            return self._finish(v, total_delta, variables_rgb, variables_tir, b,
                                identity_output)

        # ---- belief propagation ---------------------------------------------------
        h = self._h()                                              # M x N
        parity_c = parity.repeat(2, 1, 1) if parity.shape[0] == b else parity
        s = syndrome.repeat(2, 1, 1) if syndrome.shape[0] == b else syndrome
        parity_ctx = self.parity_proj(parity_c)                    # 2B x M x dim
        s_col = s.flatten(1).unsqueeze(-1)                         # 2B x M x 1

        m_cv = torch.zeros_like(v)

        for _ in range(self.iterations):
            # ---- Variable -> Check ------------------------------------------------
            m_vc_full = self.v_msg(torch.cat([branch_input, r], dim=-1))   # 2B x 256 x 768
            m_vc = torch.einsum("cv,bvd->bcd", h, m_vc_full)       # 2B x 16 x 768

            # ---- Check -> Variable ------------------------------------------------
            c_in = torch.cat([m_vc, parity_ctx, s_col.expand(-1, -1, 1)], dim=-1)
            c_out = self.c_msg(c_in)                               # 2B x 16 x 768
            m_cv = torch.einsum("cv,bcd->bvd", h, c_out)           # 2B x 256 x 768

            # ---- Update: v <- v + (1 - r) * gate * C_B(delta) ----------------------
            delta = self._step(self.update(torch.cat([branch_input, m_cv], dim=-1)))
            delta = (1.0 - r) * gate_full * delta
            v = v + delta
            total_delta = total_delta + delta
            if identity_output:
                branch_input = self.branch_norm(v)

        return self._finish(v, total_delta, variables_rgb, variables_tir, b,
                            identity_output)

    def _step(self, delta: torch.Tensor) -> torch.Tensor:
        """Zero-initialised residual projection plus an optional radial clip ``C_B``.

        ``residual_clip > 0`` bounds every step to ``B`` in L2, which is what stops the
        overshoot measured in ``tools/step_size_probe.py`` from accumulating over iterations.
        ``B`` is a fixed budget taken from a calibration quantile of the *ideal* residual
        (``tools/calibrate_residual_budget.py``), not a value tuned on the test set.
        """
        if self.residual_out is not None:
            delta = self.residual_out(delta)
        if self.residual_clip > 0:
            norm = delta.norm(dim=-1, keepdim=True)
            scale = (self.residual_clip / norm.clamp(min=1e-6)).clamp(max=1.0)
            delta = delta * scale
        if not torch.isfinite(delta).all():
            # zero the bad entries rather than propagating NaN into the tokens and the loss
            delta = torch.where(torch.isfinite(delta), delta, torch.zeros_like(delta))
            self.nonfinite_steps += 1
        return delta

    def _finish(self, v: torch.Tensor, total_delta: torch.Tensor,
                variables_rgb: torch.Tensor, variables_tir: torch.Tensor, b: int,
                identity_output: bool) -> Dict[str, torch.Tensor]:
        """Assemble the returned views for either output parameterisation.

        In ``identity_residual`` mode there is no output LayerNorm, so ``pre_norm`` *is* the
        output and the norm-only control coincides with the input: the four-cell probe
        degenerates to two cells by construction, and that has to be read as "the norm-only
        baseline no longer exists", not as "the norm-only baseline is perfect".
        """
        if identity_output:
            return {
                "corrected_rgb": v[:b], "corrected_tir": v[b:],
                "residual_rgb": total_delta[:b], "residual_tir": total_delta[b:],
                "norm_only_rgb": v[:b], "norm_only_tir": v[b:],
                "pre_norm_rgb": v[:b], "pre_norm_tir": v[b:],
                "decoder_nonfinite_steps": self.nonfinite_steps,
            }
        # norm-only control: the output LayerNorm applied to the *uncorrected* input.  It
        # isolates the pure scale change from the message updates, so a negative recovery_gain
        # can be attributed (or not) to the Tanner messages.
        pre_norm = v
        norm_only = self.out_norm(torch.cat([variables_rgb, variables_tir], dim=0))
        corrected = self.out_norm(pre_norm)
        return {
            "decoder_nonfinite_steps": self.nonfinite_steps,
            "corrected_rgb": corrected[:b],
            "corrected_tir": corrected[b:],
            "residual_rgb": total_delta[:b],
            "residual_tir": total_delta[b:],
            "norm_only_rgb": norm_only[:b],
            "norm_only_tir": norm_only[b:],
            "pre_norm_rgb": pre_norm[:b],
            "pre_norm_tir": pre_norm[b:],
        }
