"""Reliability Estimator and Target Candidate Selector -- architecture figure block 3.

    query                      B x 256 x 768
    RGB / TIR query map        B x  64 x 768
    Target Candidate Selector (Top-k Target)
    Variable Nodes V           B x 256 x 768

The reliability ``r_i in [0, 1]`` is estimated from three *independent* statistical
sources and only then handed to the decoder:

1. **template similarity** -- agreement with the trusted identity codebook;
2. **cross-modal consistency** -- do RGB and TIR agree on this token;
3. **temporal consistency** -- agreement with the trusted memory (optional).

Reliability by itself is not the contribution (many RGB-T trackers already estimate
modality quality); what matters is that it drives the syndrome-guided correction.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    a = F.normalize(a, dim=-1, eps=eps)
    b = F.normalize(b, dim=-1, eps=eps)
    return a @ b.transpose(-2, -1)


class ReliabilityEstimator(nn.Module):
    """Per-token reliability for both modalities."""

    def __init__(self, dim: int = 768, hidden: int = 128, num_identity: int = 16,
                 code_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.num_identity = num_identity
        self.code_dim = code_dim
        # identity tokens live in code_dim; project them to ``dim`` for similarity
        self.identity_key = nn.Linear(code_dim, dim)
        # three similarity cues -> scalar reliability
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_r: torch.Tensor, x_t: torch.Tensor,
                identity: torch.Tensor,
                memory: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """``x_r``/``x_t``: ``B x 256 x 768``; ``identity``: ``B x 16 x 256``."""
        # template similarity: max over the identity codebook
        keys = F.normalize(self.identity_key(identity), dim=-1)
        sim_tpl_r = _cosine(x_r, keys).max(dim=-1).values          # B x 256
        sim_tpl_t = _cosine(x_t, keys).max(dim=-1).values

        # cross-modal consistency
        sim_cross = F.cosine_similarity(x_r, x_t, dim=-1)          # B x 256

        # temporal consistency with the trusted memory (zeros when unavailable)
        if memory is not None and memory.numel() > 0:
            sim_tmp = _cosine(x_r, memory).max(dim=-1).values
        else:
            sim_tmp = torch.zeros_like(sim_cross)

        def _fuse(stpl, scross, stmp):
            cues = torch.stack([stpl, scross, stmp], dim=-1)       # B x 256 x 3
            return torch.sigmoid(self.mlp(cues).squeeze(-1) / self.temperature.abs().clamp(min=1e-3))

        r_r = _fuse(sim_tpl_r, sim_cross, sim_tmp)
        r_t = _fuse(sim_tpl_t, sim_cross, sim_tmp)
        return {
            "r_r": r_r, "r_t": r_t,                                # B x 256 each
            "sim_tpl": torch.stack([sim_tpl_r, sim_tpl_t], dim=1),  # B x 2 x 256
            "sim_cross": sim_cross,
            "sim_temporal": sim_tmp,
        }


class TargetCandidateSelector(nn.Module):
    """Score every search token by its affinity with the target codebook.

    The architecture figure labels this "Target Candidate Selector (Top-k Target)
    -> Variable Nodes V  B x 256 x 768".  With a 16x16 search grid there are exactly
    256 tokens per modality, so ``topk = 256`` retains the whole grid and the scores
    act as a **priority map** that drives

    * which tokens become graph nodes in the reliability-adaptive Tanner graph, and
    * the identity map consumed by the error locator.
    """

    def __init__(self, dim: int = 768, num_identity: int = 16, topk: int = 256,
                 code_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.topk = topk
        self.identity_proj = nn.Linear(code_dim, dim)
        self.modality_embed = nn.Parameter(torch.zeros(2, dim))
        self.score = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1),
        )

    def _score(self, x: torch.Tensor, modality: int, identity_summary: torch.Tensor,
               reference: torch.Tensor) -> torch.Tensor:
        x = x + self.modality_embed[modality]
        aff = torch.einsum("bnd,bkd->bnk", x, identity_summary).max(dim=-1).values
        feat = torch.cat([x, reference], dim=-1)                   # B x N x 1536
        return 0.5 * (aff + self.score(feat).squeeze(-1))

    def forward(self, x_r: torch.Tensor, x_t: torch.Tensor,
                identity: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, n, d = x_r.shape
        identity_summary = self.identity_proj(identity)            # B x 16 x 768
        reference = identity_summary.mean(dim=1, keepdim=True).expand(b, n, d)

        scores_rgb = self._score(x_r, 0, identity_summary, reference)
        scores_tir = self._score(x_t, 1, identity_summary, reference)
        priority = torch.maximum(scores_rgb, scores_tir)           # B x 256

        k = min(self.topk, n)
        _, index = torch.topk(priority, k=k, dim=1)                # B x k
        variables = x_r.gather(1, index.unsqueeze(-1).expand(-1, -1, d))
        return {"scores_rgb": scores_rgb, "scores_tir": scores_tir,
                "priority": priority, "index": index, "variables": variables}
