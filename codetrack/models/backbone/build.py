"""Backbone construction and OSTrack pretrained-weight loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

from .vit import SharedViTBackbone

#: Keys of the OSTrack state dict that belong to the backbone.
_BACKBONE_KEY_PREFIXES = ("patch_embed.", "pos_embed", "blocks.", "norm.",
                          "cls_token", "dist_token")

_PREFIX_STRIPPERS = ("module.", "backbone.", "backbone_0.", "net.")


def strip_prefixes(key: str, prefixes: Iterable[str] = _PREFIX_STRIPPERS) -> str:
    """Remove every known wrapper prefix from a checkpoint key."""
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key


def build_backbone(cfg: Dict[str, Any]) -> SharedViTBackbone:
    """Build the shared backbone from the ``model.backbone`` section of a config."""
    b = cfg.get("backbone", cfg)
    return SharedViTBackbone(
        search_size=b.get("img_size", 256),
        template_size=b.get("template_size", 128),
        patch_size=b.get("patch_size", 16),
        embed_dim=b.get("embed_dim", 768),
        depth=b.get("depth", 12),
        num_heads=b.get("num_heads", 12),
        pretrain_img_size=b.get("pretrain_img_size", 224),
        return_stages=b.get("return_stages", (2, 5)),
        freeze=b.get("freeze", True),
        tir_from_rgb=b.get("tir_from_rgb", True),
    )


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """Pull the parameter dict out of the various checkpoint layouts OSTrack uses."""
    if isinstance(ckpt, dict):
        for key in ("net", "model", "state_dict", "params"):
            inner = ckpt.get(key)
            if isinstance(inner, dict) and any(
                    isinstance(v, torch.Tensor) for v in inner.values()):
                return inner
    if isinstance(ckpt, dict) and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise ValueError("could not find a state dict inside the checkpoint")


def load_ostrack_pretrained(backbone: SharedViTBackbone,
                            ckpt_path: str | Path,
                            verbose: bool = True) -> Dict[str, Any]:
    """Load an OSTrack ViT-B checkpoint into :class:`SharedViTBackbone`.

    Only backbone tensors are consumed (``patch_embed`` / ``blocks`` / ``pos_embed`` /
    ``norm``); the tracking head is handled separately.  After loading:

    * the search/template position embeddings are re-derived from the freshly loaded
      ``pos_embed`` unless the checkpoint already carries them at the right shape;
    * the 1-channel TIR patch embedding is initialised from the RGB one.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"OSTrack checkpoint not found: {ckpt_path}")

    state = _extract_state_dict(torch.load(str(ckpt_path), map_location="cpu",
                                           weights_only=False))

    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        k = strip_prefixes(key)
        if not k.startswith(_BACKBONE_KEY_PREFIXES):
            continue
        if k.startswith("patch_embed."):
            # OSTrack has a single patch embed; CodeTrack keeps the RGB one.
            k = "patch_embed_rgb." + k[len("patch_embed."):]
        if k in ("dist_token", "cls_token"):
            continue  # unused by the tracker
        remapped[k] = value

    own_keys = set(backbone.state_dict().keys())
    loadable = {k: v for k, v in remapped.items() if k in own_keys}
    shape_mismatch = [k for k, v in loadable.items()
                      if v.shape != backbone.state_dict()[k].shape]
    for k in shape_mismatch:
        del loadable[k]

    missing, unexpected = backbone.load_state_dict(loadable, strict=False)

    # Position embeddings: prefer the checkpoint's own finetuned z/x grids, which were
    # trained by OSTrack, over re-deriving them from the 224x224 grid.
    has_z = "pos_embed_z" in loadable and "pos_embed_x" in loadable
    if not has_z and "pos_embed" in loadable:
        backbone.finetune_track(backbone.search_size, backbone.template_size,
                                backbone.patch_size)

    backbone.init_tir_patch_embed()

    report = {
        "checkpoint": str(ckpt_path),
        "loaded": len(loadable),
        "candidates": len(remapped),
        "shape_mismatch": sorted(shape_mismatch),
        "pos_embed_from_ckpt": bool(has_z),
        "frozen": backbone.frozen,
    }
    if verbose:
        print(f"[backbone] loaded {report['loaded']} tensors from {ckpt_path.name}")
        print(f"[backbone] search pos-embed {tuple(backbone.pos_embed_x.shape)} | "
              f"template pos-embed {tuple(backbone.pos_embed_z.shape)}")
        print(f"[backbone] frozen={backbone.frozen} | "
              f"pos_embed from ckpt={report['pos_embed_from_ckpt']}")
        if shape_mismatch:
            print(f"[backbone] skipped {len(shape_mismatch)} shape-mismatched keys: "
                  f"{shape_mismatch[:5]}")
    return report
