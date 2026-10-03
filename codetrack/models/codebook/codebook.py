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

        # search tokens -> codebook space, so that variable observations and the parity
        # reference live in the same space
        self.to_code_proj = nn.Linear(dim, code_dim)
        self.assign_temperature = nn.Parameter(torch.tensor(0.1))

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

    def parity_from_incidence(self, identity: torch.Tensor, variables: torch.Tensor,
                              H: torch.Tensor) -> torch.Tensor:
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
        """
        temp = self.assign_temperature.abs().clamp(min=1e-2)
        w = torch.softmax(
            F.normalize(variables, dim=-1) @ F.normalize(identity, dim=-1).transpose(-2, -1)
            / temp, dim=-1)                                        # B x N x K
        a_dyn = torch.einsum("mn,bnk->bmk", H, w)                 # B x M x K
        return self.norm(a_dyn @ identity)                         # B x M x d

    def connectivity(self) -> torch.Tensor:
        return self.parity_generator.connectivity()
