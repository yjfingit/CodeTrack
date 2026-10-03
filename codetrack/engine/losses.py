"""Loss terms.

The CodeTrack objective has exactly four parts::

    L = L_track + lambda_detect * L_detect + lambda_correct * L_correct
                + lambda_identity * L_identity

with the reference weights ``lambda_detect = 1.0``, ``lambda_correct = 2.0``,
``lambda_identity = 0.1``:

* ``L_track``    -- OSTrack-style Focal + L1 + GIoU.  Keeps the tracking task alive
  while the correction machinery is trained.
* ``L_detect``   -- supervision for reliability / syndrome / error localization.
  The model must *find* the corrupted tokens, not merely survive them: without this
  term the syndrome could collapse to a constant and "detection" would be meaningless.
* ``L_correct``  -- the belief-propagation decoder's output must approach the
  corruption-free token.  This is the core ability (highest weight), so it is
  supervised directly against the clean teacher tokens rather than only through
  the tracking loss.
* ``L_identity`` -- the repaired token must still belong to the *original* target.
  Implemented as a distillation over the identity codebook, which stops the decoder
  from "repairing" a token onto a look-alike distractor (another person, a similar
  vehicle) instead of the tracked object.

The clean teacher costs one extra pass through the CodeTrack modules; the frozen
backbone is shared and executed only once.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_heatmap(size: int, cx: torch.Tensor, cy: torch.Tensor,
                     sigma: float = 0.05) -> torch.Tensor:
    """``B`` normalised centres -> ``B x 1 x size x size`` Gaussian targets."""
    dev = cx.device
    xs = (torch.arange(size, dtype=torch.float32, device=dev).view(1, -1) + 0.5) / size
    ys = (torch.arange(size, dtype=torch.float32, device=dev).view(-1, 1) + 0.5) / size
    gx = xs.unsqueeze(0) - cx.unsqueeze(-1).unsqueeze(-1)
    gy = ys.unsqueeze(0) - cy.unsqueeze(-1).unsqueeze(-1)
    return torch.exp(-(gx ** 2 + gy ** 2) / (2 * sigma ** 2)).unsqueeze(1)


def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25,
               gamma: float = 2.0) -> torch.Tensor:
    """Binary focal loss on the centre heatmap."""
    pred = pred.clamp(1e-4, 1 - 1e-4)
    ce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    p_t = target * pred + (1 - target) * (1 - pred)
    alpha_t = target * alpha + (1 - target) * (1 - alpha)
    return (alpha_t * (1 - p_t).pow(gamma) * ce).mean()


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``pred``/``target``: ``B x 4`` in ``cx, cy, w, h`` (normalised)."""
    p = _cxcywh_to_xyxy(pred.clamp(1e-5, 1.0))
    t = _cxcywh_to_xyxy(target.clamp(1e-5, 1.0))

    inter_x1 = torch.max(p[:, 0], t[:, 0])
    inter_y1 = torch.max(p[:, 1], t[:, 1])
    inter_x2 = torch.min(p[:, 2], t[:, 2])
    inter_y2 = torch.min(p[:, 3], t[:, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area_p = (p[:, 2] - p[:, 0]).clamp(min=0) * (p[:, 3] - p[:, 1]).clamp(min=0)
    area_t = (t[:, 2] - t[:, 0]).clamp(min=0) * (t[:, 3] - t[:, 1]).clamp(min=0)
    union = area_p + area_t - inter + 1e-6

    iou = inter / union
    c_x1 = torch.min(p[:, 0], t[:, 0])
    c_y1 = torch.min(p[:, 1], t[:, 1])
    c_x2 = torch.max(p[:, 2], t[:, 2])
    c_y2 = torch.max(p[:, 3], t[:, 3])
    area_c = (c_x2 - c_x1).clamp(min=0) * (c_y2 - c_y1).clamp(min=0) + 1e-6
    giou = iou - (area_c - union) / area_c
    return (1.0 - giou).mean()


class CodeTrackLoss(nn.Module):
    """The four-term CodeTrack objective (see the module docstring)."""

    def __init__(self, cls_weight: float = 1.0, l1_weight: float = 5.0,
                 giou_weight: float = 2.0, fetal_alpha: float = 0.25,
                 focal_gamma: float = 2.0, lambda_detect: float = 1.0,
                 lambda_correct: float = 2.0, lambda_identity: float = 0.1,
                 sigma: float = 0.05, num_parity: int = 16, dim: int = 768,
                 code_dim: int = 256, temperature: float = 0.1):
        super().__init__()
        self.cls_weight = cls_weight
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight
        self.focal_alpha = fetal_alpha
        self.focal_gamma = focal_gamma
        self.lambda_detect = lambda_detect
        self.lambda_correct = lambda_correct
        self.lambda_identity = lambda_identity
        self.sigma = sigma
        self.num_parity = num_parity
        self.temperature = temperature
        # projects the identity codebook (code_dim) into token space (dim)
        self.identity_proj = nn.Linear(code_dim, dim)

    # ------------------------------------------------------------------ pieces
    def _identity_dist(self, tokens: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Soft assignment of every token over the identity codebook."""
        sim = F.normalize(tokens, dim=-1) @ keys.transpose(-2, -1)     # B x N x K
        return F.softmax(sim / self.temperature, dim=-1)

    def _detect_loss(self, outputs: Dict[str, torch.Tensor],
                     mask: torch.Tensor) -> torch.Tensor:
        """Reliability + syndrome + localization supervision (fp32)."""
        mask = mask.float()
        target_reliable = (1.0 - mask).clamp(0.0, 1.0)
        r_r = outputs["reliability_rgb"].float().clamp(1e-4, 1 - 1e-4)
        r_t = outputs["reliability_tir"].float().clamp(1e-4, 1 - 1e-4)
        reliability = (F.binary_cross_entropy(r_r, target_reliable)
                       + F.binary_cross_entropy(r_t, target_reliable))

        # a check that watches a corrupted token must fire
        b, n = mask.shape
        per_check = n // max(self.num_parity, 1)
        check_target = mask.view(b, self.num_parity, per_check).mean(dim=-1).clamp(0.0, 1.0)
        syndrome = outputs["syndrome"].float().flatten(1).clamp(1e-4, 1 - 1e-4)
        syndrome_loss = F.binary_cross_entropy(syndrome, check_target)

        # the locator is a per-token corruption probability (averaged over graph nodes).
        # Kept as a bounded BCE: the earlier log-likelihood form exploded to ~1e3 and
        # dominated the whole objective.
        locator = outputs["locator"].float().mean(dim=1).clamp(1e-4, 1 - 1e-4)   # B x N
        localize = F.binary_cross_entropy(locator, mask)

        return reliability + syndrome_loss + localize

    # ------------------------------------------------------------------ forward
    def forward(self, outputs: Dict[str, torch.Tensor],
                target_box: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``outputs``: the dict from :class:`CodeTrack`; ``target_box``: ``B x 4``
        normalised ``(cx, cy, w, h)``."""
        device = target_box.device
        target_box = target_box.float()
        parts: Dict[str, torch.Tensor] = {}

        # ---- L_track ----------------------------------------------------------
        score_map = outputs["score_map_ctr"].float()              # B x 1 x 16 x 16
        gt_heat = gaussian_heatmap(score_map.shape[-1], target_box[:, 0], target_box[:, 1],
                                   self.sigma)
        cls = focal_loss(score_map, gt_heat, self.focal_alpha, self.focal_gamma)
        bbox = outputs["bbox"].float()
        l1 = F.l1_loss(bbox[:, 2:], target_box[:, 2:])
        giou = giou_loss(bbox, target_box)
        track = self.cls_weight * cls + self.l1_weight * l1 + self.giou_weight * giou

        # ---- L_detect ---------------------------------------------------------
        detect = torch.zeros((), device=device)
        mask = outputs.get("token_mask")
        if mask is not None and float(mask.sum()) > 0:
            detect = self._detect_loss(outputs, mask.to(device))

        # ---- L_correct / L_identity ------------------------------------------
        correct = torch.zeros((), device=device)
        identity = torch.zeros((), device=device)
        clean = outputs.get("clean_tokens")
        if clean is not None:
            if self.lambda_correct > 0:
                correct = (F.l1_loss(outputs["corrected_rgb"].float(), clean["rgb"].float())
                           + F.l1_loss(outputs["corrected_tir"].float(), clean["tir"].float()))
            if self.lambda_identity > 0:
                keys = F.normalize(self.identity_proj(outputs["identity_tokens"].float()), dim=-1)
                with torch.no_grad():
                    target_dist = self._identity_dist(clean["rgb"].float(), keys)
                pred_dist = self._identity_dist(outputs["corrected_rgb"].float(), keys)
                identity = -(target_dist * torch.log(pred_dist.clamp(min=1e-8))).sum(-1).mean()

        parts.update({"cls": cls.detach(), "l1": l1.detach(), "giou": giou.detach(),
                      "detect": detect.detach(), "correct": correct.detach(),
                      "identity": identity.detach()})

        total = (track
                 + self.lambda_detect * detect
                 + self.lambda_correct * correct
                 + self.lambda_identity * identity)
        parts["track"] = track.detach()
        parts["loss"] = total
        return parts
