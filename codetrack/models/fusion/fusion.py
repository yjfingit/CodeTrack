"""Fusion -- architecture figure block 7.

    Corrected zg                B x 256 x 768
    Corrected xg                B x 256 x 768
    Residual                    B x 256 x 768
    Reshape to Feature Map      B x 768 x 16 x 16
    FPN fusion (block 2 / block 5)
        RGB: B x 512 x 16 x 16
        TIR: B x 512 x 16 x 16

Per-modality FPN is exactly 512 channels because block-2 and block-5 taps are each
projected to 256 and concatenated (256 + 256 = 512).  The two modalities are then
fused back to 768 so the tracking head can reuse the OSTrack head weights.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _tokens_to_map(tokens: torch.Tensor, grid: int) -> torch.Tensor:
    """``B x (grid*grid) x C`` -> ``B x C x grid x grid``."""
    b, n, c = tokens.shape
    return tokens[:, -grid * grid:, :].transpose(1, 2).reshape(b, c, grid, grid).contiguous()


class CodeTrackFusion(nn.Module):
    """Token residual fusion + FPN over backbone taps."""

    def __init__(self, dim: int = 768, fpn_dim: int = 256, head_dim: int = 768,
                 grid: int = 16, taps: Tuple[int, ...] = (2, 5)):
        super().__init__()
        self.dim = dim
        self.grid = grid
        self.taps = tuple(taps)

        self.tap_proj = nn.Linear(dim, fpn_dim)

        # fuse the 2 modalities x (fpn_dim * n_taps) into head_dim
        fused_in = 2 * fpn_dim * len(self.taps)
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_in, head_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_dim, head_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_dim),
        )
        self.token_proj = nn.Conv2d(dim, head_dim, kernel_size=1, bias=False)

    def _modality_fpn(self, taps: Dict[str, torch.Tensor]) -> torch.Tensor:
        """``{'block2': B x 320 x 768, ...}`` -> ``B x 512 x 16 x 16``."""
        maps = []
        for key in (f"block{i}" for i in self.taps):
            if key not in taps:
                raise KeyError(f"missing backbone tap {key!r}; got {sorted(taps)}")
            feat = self.tap_proj(taps[key])
            maps.append(_tokens_to_map(feat, self.grid))
        return torch.cat(maps, dim=1)                     # B x (256*n_taps) x 16 x 16

    def forward(self, corrected_rgb: torch.Tensor, corrected_tir: torch.Tensor,
                residual_rgb: torch.Tensor, residual_tir: torch.Tensor,
                taps_r: Dict[str, torch.Tensor], taps_t: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        # token-level fusion: corrected + residual, averaged over modalities
        tokens = 0.5 * (corrected_rgb + corrected_tir) + 0.5 * (residual_rgb + residual_tir)
        feat_map = _tokens_to_map(tokens, self.grid)                      # B x 768 x 16 x 16
        token_map = self.token_proj(feat_map)

        fpn_r = self._modality_fpn(taps_r)                                # B x 512 x 16 x 16
        fpn_t = self._modality_fpn(taps_t)
        fpn = self.fuse(torch.cat([fpn_r, fpn_t], dim=1))                 # B x 768 x 16 x 16

        out = token_map + fpn
        return {"feature_map": out, "fpn_rgb": fpn_r, "fpn_tir": fpn_t,
                "token_map": token_map}
