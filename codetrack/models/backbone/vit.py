"""Shared ViT-B/16 backbone for CodeTrack.

Ported from OSTrack (Apache-2.0), https://github.com/botaoye/OSTrack:
``lib/models/ostrack/vit.py`` and ``lib/models/ostrack/base_backbone.py``.

CodeTrack-specific changes (everything else is kept bit-identical, so the public
OSTrack checkpoint loads without adaptation):

1. **Two patch embeddings, one shared transformer stack.**  The architecture figure
   draws a separate ``Patch Embed (16x16)`` per modality but a single
   ``Shared ViT-B/16 (12 Blocks)``.  RGB keeps the pretrained 3-channel projection;
   TIR gets a 1-channel projection initialised as the channel-mean of the RGB one.
2. **Optional freezing** -- the figure marks the transformer blocks as *Frozen*, so
   only the CodeTrack modules around the backbone are trained.
3. **Intermediate taps** on block 2 / block 5 for FPN fusion.
4. **Absolute position embeddings** (``Abs. Pos``) exactly as in OSTrack.  The figure
   also mentions RoPE; OSTrack's ViT-B is absolute-position based, and reusing its
   weights takes priority, so RoPE is intentionally not used here.

Tensor shapes with the reference configuration (search 256, template 128, patch 16):

    template   B x 3 x 128 x 128  -> patch embed -> B x  64 x 768
    search     B x 3 x 256 x 256  -> patch embed -> B x 256 x 768
    concatenated (template, search)                -> B x 320 x 768
"""

from __future__ import annotations

from functools import partial
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # timm >= 0.9
    from timm.layers import DropPath, Mlp, trunc_normal_
except ImportError:  # pragma: no cover - legacy timm
    from timm.models.layers import DropPath, Mlp, trunc_normal_

from .patch_embed import PatchEmbed


class Attention(nn.Module):
    """Multi-head self attention, identical to OSTrack / timm ViT."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x


class Block(nn.Module):
    """Pre-norm transformer block, identical to OSTrack."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            feat, attn = self.attn(self.norm1(x), True)
            x = x + self.drop_path(feat)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def combine_tokens(template_tokens: torch.Tensor, search_tokens: torch.Tensor) -> torch.Tensor:
    """``[Z, X]`` concatenation -- OSTrack's ``cat_mode='direct'``."""
    return torch.cat((template_tokens, search_tokens), dim=1)


def recover_tokens(merged_tokens: torch.Tensor, len_template: int, len_search: int) -> torch.Tensor:
    """Identity for ``cat_mode='direct'``."""
    return merged_tokens


class SharedViTBackbone(nn.Module):
    """ViT-B/16 with a shared transformer stack and per-modality patch embedding."""

    def __init__(self,
                 search_size: int = 256,
                 template_size: int = 128,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 pretrain_img_size: int = 224,
                 return_stages: Iterable[int] = (2, 5),
                 freeze: bool = True,
                 tir_from_rgb: bool = True):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.search_size = search_size
        self.template_size = template_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.pretrain_img_size = pretrain_img_size
        self.return_stages = tuple(return_stages)
        self.tir_from_rgb = tir_from_rgb
        self.frozen = False

        # ---- per-modality patch embedding (RGB 3ch keeps the pretrained weights) ----
        self.patch_embed_rgb = PatchEmbed(img_size=pretrain_img_size, patch_size=patch_size,
                                          in_chans=3, embed_dim=embed_dim)
        self.patch_embed_tir = PatchEmbed(img_size=pretrain_img_size, patch_size=patch_size,
                                          in_chans=1, embed_dim=embed_dim)

        # ---- shared transformer stack ----
        num_patches = (pretrain_img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # LayerNorms applied to the FPN taps (kept trainable, they are CodeTrack additions)
        self.inter_norms = nn.ModuleDict({str(i): norm_layer(embed_dim) for i in self.return_stages})

        self.finetune_track(search_size, template_size, patch_size)
        self.init_weights()

        if freeze:
            self.freeze_backbone()

    # ------------------------------------------------------------------ init
    def init_weights(self) -> None:
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_vit_weights)

    @staticmethod
    def _init_vit_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    # ------------------------------------------------------- position embedding
    def finetune_track(self, search_size: int, template_size: int, patch_size: int = 16) -> None:
        """Derive the search/template absolute position embeddings from ``pos_embed``.

        Mirrors ``BaseBackbone.finetune_track`` from OSTrack: the pretrained 224x224
        grid (14x14 patches) is bicubically resized to 16x16 (search, 256) and
        8x8 (template, 128).
        """
        patch_pos_embed = self.pos_embed[:, 1:, :]                 # drop the cls token slot
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, _ = patch_pos_embed.shape
        p = self.pretrain_img_size // self.patch_size
        patch_pos_embed = patch_pos_embed.reshape(B, E, p, p)

        def _resize(size):
            grid = F.interpolate(patch_pos_embed, size=size, mode="bicubic", align_corners=False)
            return nn.Parameter(grid.flatten(2).transpose(1, 2))

        self.pos_embed_z = _resize((template_size // patch_size, template_size // patch_size))
        self.pos_embed_x = _resize((search_size // patch_size, search_size // patch_size))

    # ------------------------------------------------------------------ freeze
    def freeze_backbone(self) -> None:
        """Freeze every parameter of the backbone, including the FPN tap LayerNorms.

        The figure marks the transformer as *Frozen*; the tap norms were CodeTrack
        additions, but leaving 0.003 M trainable inside a "frozen" backbone is exactly
        the kind of detail a reviewer pokes at, so they are frozen too.
        """
        for module in (self.patch_embed_rgb, self.patch_embed_tir, self.blocks,
                       self.norm, self.inter_norms):
            for p in module.parameters():
                p.requires_grad = False
        for p in (self.pos_embed, self.pos_embed_z, self.pos_embed_x):
            p.requires_grad = False
        self.frozen = True

    def unfreeze_backbone(self) -> None:
        for module in (self.patch_embed_rgb, self.patch_embed_tir, self.blocks,
                       self.norm, self.inter_norms):
            for p in module.parameters():
                p.requires_grad = True
        for p in (self.pos_embed, self.pos_embed_z, self.pos_embed_x):
            p.requires_grad = True
        self.frozen = False

    def init_tir_patch_embed(self) -> None:
        """Initialise the 1-channel TIR projection as the channel-mean of the RGB one.

        A thermal frame is (approximately) the luminance of the visible frame, so the
        mean over the RGB input channels is the natural single-channel counterpart of
        the pretrained 3-channel filter.
        """
        if not self.tir_from_rgb:
            return
        with torch.no_grad():
            rgb_w = self.patch_embed_rgb.proj.weight
            self.patch_embed_tir.proj.weight.copy_(rgb_w.mean(dim=1, keepdim=True))
            self.patch_embed_tir.proj.bias.copy_(self.patch_embed_rgb.proj.bias)

    # ----------------------------------------------------------------- forward
    def forward_stream(self, template: torch.Tensor, search: torch.Tensor,
                       modality: str = "rgb", return_inter: bool = True):
        """Run one modality through the shared stack.

        Returns ``(z, x, inter)`` with ``z`` being the template tokens and ``x`` the
        search tokens, both ``B x L x 768``.
        """
        if modality == "rgb":
            embed = self.patch_embed_rgb
        elif modality == "tir":
            embed = self.patch_embed_tir
        else:
            raise ValueError(f"unknown modality {modality!r}")

        z = embed(template) + self.pos_embed_z
        x = embed(search) + self.pos_embed_x

        tokens = combine_tokens(z, x)                              # B x (Lz+Lx) x C
        tokens = self.pos_drop(tokens)

        inter: Dict[str, torch.Tensor] = {}
        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens)
            if return_inter and i in self.return_stages:
                inter[f"block{i}"] = self.inter_norms[str(i)](tokens)

        tokens = self.norm(tokens)
        tokens = recover_tokens(tokens, self.pos_embed_z.shape[1], self.pos_embed_x.shape[1])

        lens_z = self.pos_embed_z.shape[1]
        return tokens[:, :lens_z], tokens[:, lens_z:], inter

    def forward(self, rgb: Tuple[torch.Tensor, torch.Tensor],
                tir: Tuple[torch.Tensor, torch.Tensor],
                return_inter: bool = True) -> Dict[str, torch.Tensor]:
        """``rgb``/``tir`` are ``(template, search)`` pairs."""
        z_r, x_r, inter_r = self.forward_stream(rgb[0], rgb[1], modality="rgb",
                                                return_inter=return_inter)
        z_t, x_t, inter_t = self.forward_stream(tir[0], tir[1], modality="tir",
                                                return_inter=return_inter)
        return {"z_r": z_r, "x_r": x_r, "z_t": z_t, "x_t": x_t,
                "inter_r": inter_r, "inter_t": inter_t}
