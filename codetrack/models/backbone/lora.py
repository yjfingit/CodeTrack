"""Low-rank adaptation injected into the *frozen* backbone.

Why this exists
---------------
The round-1/2 review made the point that "freeze the backbone" and "do not adapt the backbone"
are different things.  ViPT and SDSTrack freeze the pretrained weights and still inject trainable
capacity *inside* every block (prompts, adapters); this repository instead keeps the whole feature
extractor fixed and trains 26.5M of parameters entirely *after* it.  Those are not the same
configuration, and the difference matters for the paper's central claim: if the per-token repair
module shows no benefit, one live explanation is that the representation it operates on cannot be
adapted at all, and no experiment so far distinguishes that from "the module is useless".

`LoRAQKV` makes the distinction testable with one variable: the same frozen weights, the same
budget, the same recipe, with a small trainable low-rank update added to the attention queries and
values.

Design notes
------------
* **Slices, not a new projection.**  OSTrack uses a fused ``qkv`` linear, so the delta is computed
  for ``q`` and ``v`` separately and written into the corresponding slices of a zero tensor.  A
  single LoRA on the fused projection would also perturb ``k``, which then changes the attention
  logits multiplicatively -- a different intervention.
* **``B`` is zero-initialised**, so the module is an exact identity at step 0 and the adapted arm
  starts bit-identical to the frozen one.  ``A`` is random: an all-zero initialisation would make
  the gradient through ``A`` zero on the first step.
* **Per modality by default.**  The two streams share the backbone weights, so a *shared* LoRA
  would apply the same update to both; ``per_modality=True`` routes by the ``modality`` argument
  that ``SharedViTBackbone.forward_stream`` already carries.  Sharing is available because it is
  the cheaper arm and the comparison is cheap to make.
* ``freeze_backbone()`` must not freeze these parameters, and ``unfreeze_backbone()`` must not
  silently start training the pretrained tensors either -- both are covered by tests.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class LoRAQKV(nn.Module):
    """Frozen fused ``qkv`` linear plus trainable low-rank deltas on the Q and V slices.

    ``base`` holds the original (frozen) parameters, so the state dict layout is the same one
    OSTrack checkpoints use and loading is unchanged.
    """

    TARGET_SLICES = {"q": 0, "v": 2}

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0,
                 targets=("q", "v"), per_modality: bool = True,
                 modalities=("rgb", "tir"), seed: int = 0,
                 generator: Optional[torch.Generator] = None):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        unknown = set(targets) - set(self.TARGET_SLICES)
        if unknown:
            raise ValueError(f"unknown LoRA targets {sorted(unknown)}; "
                             f"supported: {sorted(self.TARGET_SLICES)}")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.dim = base.out_features // 3
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.targets = tuple(targets)
        self.per_modality = bool(per_modality)
        self.modalities = tuple(modalities)

        variants = self.modalities if self.per_modality else ("shared",)
        # A **local** generator, on purpose.  Drawing the adapter's init from the global RNG would
        # shift every later draw, so the CodeTrack modules built after the backbone would get
        # different initial weights in the adapted arm than in the frozen one.  Then F0/F1/A0/A1
        # would differ in two things instead of one, which is exactly what the capacity-location
        # comparison has to avoid (docs/results.md 6.27.4).
        gen = generator if generator is not None else torch.Generator().manual_seed(int(seed))
        self.lora_a = nn.ParameterDict()
        self.lora_b = nn.ParameterDict()
        for variant in variants:
            for target in self.targets:
                a = nn.Parameter(torch.empty(self.rank, self.in_features))
                b = nn.Parameter(torch.zeros(self.dim, self.rank))
                # ``B = 0`` makes the module an exact identity at step 0; ``A`` must not be zero or
                # the gradient through it would vanish on the first step.
                nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=gen)
                self.lora_a[f"{variant}_{target}"] = a
                self.lora_b[f"{variant}_{target}"] = b

    def variant(self, modality: Optional[str]) -> str:
        if not self.per_modality:
            return "shared"
        if modality not in self.modalities:
            raise ValueError(f"LoRA is per-modality but got modality={modality!r}; "
                             f"expected one of {self.modalities}")
        return str(modality)

    def delta(self, x: torch.Tensor, modality: Optional[str] = None) -> torch.Tensor:
        """The additive qkv update, zero except in the Q and V slices."""
        variant = self.variant(modality)
        delta = torch.zeros(x.shape[:-1] + (self.out_features,), dtype=x.dtype,
                            device=x.device)
        for target in self.targets:
            a = self.lora_a[f"{variant}_{target}"]
            b = self.lora_b[f"{variant}_{target}"]
            update = (x @ a.transpose(0, 1)) @ b.transpose(0, 1)      # ... x dim
            start = self.TARGET_SLICES[target] * self.dim
            delta[..., start:start + self.dim] = delta[..., start:start + self.dim] \
                + update * self.scaling
        return delta

    def forward(self, x: torch.Tensor, modality: Optional[str] = None) -> torch.Tensor:
        return self.base(x) + self.delta(x, modality)

    def extra_repr(self) -> str:
        return (f"rank={self.rank}, alpha={self.alpha}, targets={self.targets}, "
                f"per_modality={self.per_modality}")


class NullLoRA(nn.Module):
    """The disabled form: identical arithmetic, no extra state, so the default path is untouched."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor, modality: Optional[str] = None) -> torch.Tensor:
        return self.base(x)
