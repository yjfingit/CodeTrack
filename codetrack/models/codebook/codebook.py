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

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import trunc_normal_
except ImportError:  # pragma: no cover
    from timm.models.layers import trunc_normal_


class SparseParityGenerator(nn.Module):
    """``P = A @ U`` with a learnable, structurally sparse ``A``.

    The *support* of ``A`` is a fixed random pattern (``links_per_row`` ones per row,
    ``A_ij in {0, 1}``); only the weights on that support are trained, giving a soft
    sparse ``A_ij in [0, 1]``.  Sparsity is what makes error *localization* possible
    later: a small number of failing checks pinpoints the corrupt token.
    """

    def __init__(self, num_identity: int = 16, num_parity: int = 16,
                 links_per_row: int = 4, generator: Optional[torch.Generator] = None):
        super().__init__()
        self.num_identity = num_identity
        self.num_parity = num_parity
        self.links_per_row = min(max(links_per_row, 1), num_identity)

        pattern = torch.zeros(num_parity, num_identity)
        for row in range(num_parity):
            idx = torch.randperm(num_identity, generator=generator)[:self.links_per_row]
            pattern[row, idx] = 1.0
        self.register_buffer("support", pattern, persistent=True)

        weight = pattern.clone()
        weight = weight / weight.sum(dim=1, keepdim=True).clamp(min=1.0)
        self.A = nn.Parameter(weight)

    def forward(self, identity: torch.Tensor) -> torch.Tensor:
        """``identity``: ``B x K x d`` -> ``B x M x d``."""
        a = self.A * self.support
        a = a / a.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return a @ identity

    def connectivity(self) -> torch.Tensor:
        """Binary support of ``A`` -- the incidence structure of the Tanner graph."""
        return (self.support > 0).float()


class TargetCodebookEncoder(nn.Module):
    """Produce the target error-correcting codebook ``C_t = {U_t, P_t}``."""

    def __init__(self, dim: int = 768, code_dim: int = 256, proj_dim: int = 512,
                 num_identity: int = 16, num_parity: int = 16, num_heads: int = 8,
                 links_per_row: int = 4, dropout: float = 0.0):
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
        self.parity_generator = SparseParityGenerator(
            num_identity=num_identity, num_parity=num_parity, links_per_row=links_per_row)

        self.norm = nn.LayerNorm(code_dim)
        trunc_normal_(self.identity_queries, std=.02)

    def forward(self, z_r: torch.Tensor, z_t: torch.Tensor,
                memory: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(identity U, parity P, query)``.

        ``U``: ``B x 16 x 256``, ``P``: ``B x 16 x 256``, ``query``: ``B x 144 x 768``.
        """
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

        identity = self.ecc_identity(q[:, :self.num_identity])   # B x 16 x 256
        parity = self.parity_generator(identity)                 # B x 16 x 256
        return identity, self.norm(parity), q

    def connectivity(self) -> torch.Tensor:
        return self.parity_generator.connectivity()
