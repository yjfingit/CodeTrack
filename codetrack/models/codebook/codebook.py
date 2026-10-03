"""Target Codebook Encoder -- architecture figure block 2.

Ported/derived from the CodeTrack architecture figure:

    (Z_R, Z_T, M_{t-1})   B x [64, 64, 128] x 768
        fc               -> B x 128 x 512
        Linear           -> B x 128 x 512
        Query            -> B x 144 x 768
        Identity Tokens U  B x 16 x 256
        Sparse Parity Tokens P  B x 16 x 256
        ECC Projection    Linear 768 -> 256

Design notes
------------
* The codebook is built **only from trusted information** (initial template of both
  modalities, optionally the trusted memory).  Parity must never be derived from the
  current -- possibly corrupted -- search frame, otherwise the redundancy carries no
  coding power.
* ``144 = 128 + 16``: the 128 projected template tokens are concatenated with 16
  learnable identity queries; a transformer block lets the queries attend to the
  template, then the first 16 outputs become the identity tokens.
* Identity tokens are projected to ``d_code = 256`` by the ECC projection; parity
  tokens are produced by a **learnable sparse matrix** ``A`` (Tanner-graph style),
  ``P = A @ U``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import trunc_normal_
except ImportError:  # pragma: no cover
    from timm.models.layers import trunc_normal_


class TargetCodebookEncoder(nn.Module):
    """Produce the target error-correcting codebook ``C_t = {U_t, P_t}``."""

    def __init__(self, dim: int = 768, code_dim: int = 256, proj_dim: int = 512,
                 num_identity: int = 16, num_parity: int = 16, num_heads: int = 8,
                 links_per_row: int = 4, dropout: float = 0.0,
                 temperature: float = 0.1, tau_min: float = 0.02,
                 tau_max: float = 0.5):
        super().__init__()
        self.dim = dim
        self.code_dim = code_dim
        self.num_identity = num_identity
        self.num_parity = num_parity

        # fc + Linear: 768 -> 512 -> 512 (figure labels both as B x 128 x 512)
        self.fc = nn.Linear(dim, proj_dim)
        self.fc2 = nn.Linear(proj_dim, proj_dim)

        # 16 learnable identity queries appended to the 128 template tokens
        self.identity_queries = nn.Parameter(torch.zeros(1, num_identity, proj_dim))
        self.proj = nn.Linear(proj_dim, dim)          # -> Query, B x 144 x 768

        self.codebook_block = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )

        # ECC projection: Linear 768 -> 256
        self.ecc_identity = nn.Linear(dim, code_dim)

        # search tokens -> codebook space, so that variable observations and the parity
        # reference live in the same space
        self.to_code_proj = nn.Linear(dim, code_dim)

        # Bounded learnable assignment temperature.  Previously this was
        # ``abs().clamp(min=1e-2)`` on a raw parameter, which can sit at 0 (assignments
        # collapse onto one identity codeword) or explode (assignments become uniform and
        # the parity stops being a weighted recombination).  A sigmoid inside [tau_min,
        # tau_max] rules both out while staying differentiable everywhere.
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        # init so that sigmoid(logit) reproduces the reference temperature 0.1
        frac = (temperature - tau_min) / max(tau_max - tau_min, 1e-6)
        frac = min(max(frac, 1e-3), 1 - 1e-3)
        self.temperature_logit = nn.Parameter(torch.tensor(math.log(frac / (1 - frac))))

        self.norm = nn.LayerNorm(code_dim)
        trunc_normal_(self.identity_queries, std=.02)

    def encode_identity(self, z_r: torch.Tensor, z_t: torch.Tensor,
                        memory: Optional[torch.Tensor] = None
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Template -> identity codewords ``U`` (``B x 16 x 256``) and the query tensor."""
        b = z_r.shape[0]
        tokens = [z_r, z_t]
        if memory is not None and memory.numel() > 0:
            tokens.append(memory)
        h = torch.cat(tokens, dim=1)                     # B x 128 x 768

        h = self.fc2(F.gelu(self.fc(h)))                 # B x 128 x 512
        queries = self.identity_queries.expand(b, -1, -1)
        q = torch.cat([h, queries], dim=1)               # B x 144 x 512
        q = self.proj(q)                                 # B x 144 x 768  (Query)
        q = self.codebook_block(q)                       # B x 144 x 768

        # the 16 learnable queries sit AFTER the 128 template tokens
        identity = self.ecc_identity(q[:, -self.num_identity:])   # B x 16 x 256
        return identity, q

    def to_code(self, tokens: torch.Tensor) -> torch.Tensor:
        """Project search tokens into the codebook space, ``B x N x 256``."""
        return self.to_code_proj(tokens)

    def temperature(self) -> torch.Tensor:
        """Bounded learnable ``tau`` in ``[tau_min, tau_max]``.

        ``tau = tau_min + (tau_max - tau_min) * sigmoid(rho)``: it can never reach 0
        (all variables collapsing onto a single identity codeword) nor blow up (every
        variable assigned equally, i.e. the parity stops being a weighted recombination of
        the codebook).  Both extremes silently destroy the coding meaning of the parity.
        """
        return self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.temperature_logit)

    def assignment(self, variables: torch.Tensor, identity: torch.Tensor,
                   reliability: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Soft assignment ``w`` of every variable over the identity codebook.

        ``reliability`` (``B x N``, in ``[0, 1]``) down-weights the *observed* token
        towards its neighbourhood mean before the cosine is taken.  This is the cheap
        version of "the assignment must not depend on a corrupted received word": a token
        the model already distrusts cannot drag the mixture coefficients around, so the
        parity stays a function of the **trusted template** plus a mild observation cue.
        A fully template-only assignment would be corruption-proof but would also be
        frame-independent, which throws away the only adaptivity the check has.
        """
        v = F.normalize(variables, dim=-1)
        if reliability is not None:
            r = reliability.to(v.dtype).unsqueeze(-1)
            # keep the (reliable) mean direction of the neighbourhood as a stand-in
            v = F.normalize(r * v + (1.0 - r) * v.mean(dim=1, keepdim=True), dim=-1)
        cos = v @ F.normalize(identity, dim=-1).transpose(-2, -1)     # B x N x K
        return torch.softmax(cos / self.temperature(), dim=-1)

    def parity_from_incidence(self, identity: torch.Tensor, variables: torch.Tensor,
                              H: torch.Tensor,
                              reliability: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reference codewords for the checks defined by ``H``.

        ``w[i, m]`` measures how strongly variable ``i`` is explained by identity token
        ``m`` (soft assignment in ``[0, 1]`` with rows summing to 1).  The induced parity
        matrix is ``A = H @ w`` -- shape ``M x K`` -- and

            parity_j = sum_m A[j, m] U_m

        is therefore the value check ``j`` *expects* over exactly the neighbourhood it
        watches.  This is what binds the codebook to the Tanner incidence: without it the
        parity tokens and ``H`` are two unrelated structures and the syndrome compares a
        search observation against an arbitrary reference.

        Args:
            identity: ``B x K x d`` template-derived codewords.
            variables: ``B x N x dim`` search tokens (fused over modalities by caller).
            H: ``M x N`` normalised parity-check matrix from the decoder.
            reliability: optional ``B x N`` reliability used to stabilise ``w``.
        """
        w = self.assignment(variables, identity, reliability=reliability)
        a_dyn = torch.einsum("mn,bnk->bmk", H, w)                 # B x M x K
        return self.norm(a_dyn @ identity)                         # B x M x d
