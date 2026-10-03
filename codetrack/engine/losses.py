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
                     mask_rgb: torch.Tensor, mask_tir: torch.Tensor) -> torch.Tensor:
        """Reliability + syndrome + localization supervision (fp32).

        The syndrome target is derived from the **shared parity-check matrix**: a check
        must fire iff it actually watches at least one corrupted token.  RGB and TIR are
        supervised with their own masks, so a healthy modality is not dragged down.
        """
        mask_rgb = mask_rgb.float()
        mask_tir = mask_tir.float()
        mask_any = torch.clamp(mask_rgb + mask_tir, max=1.0)

        reliability = (
            F.binary_cross_entropy(outputs["reliability_rgb"].float().clamp(1e-4, 1 - 1e-4),
                                   (1.0 - mask_rgb).clamp(0.0, 1.0))
            + F.binary_cross_entropy(outputs["reliability_tir"].float().clamp(1e-4, 1 - 1e-4),
                                     (1.0 - mask_tir).clamp(0.0, 1.0)))

        h = outputs["H"].float()                                    # M x N
        check_target = ((h @ mask_any.t()) > 0).float().t()         # B x M
        syndrome = outputs["syndrome"].float().flatten(1).clamp(1e-4, 1 - 1e-4)
        syndrome_loss = F.binary_cross_entropy(syndrome, check_target)

        locator = outputs["locator_scattered"].float().clamp(1e-4, 1 - 1e-4)
        localize = F.binary_cross_entropy(locator, mask_any)

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
        mask_r = outputs.get("token_mask_rgb")
        mask_t = outputs.get("token_mask_tir")
        has_mask = mask_r is not None and mask_t is not None
        if has_mask and float(mask_r.sum() + mask_t.sum()) > 0:
            detect = self._detect_loss(outputs, mask_r.to(device), mask_t.to(device))

        # ---- L_correct / L_identity ------------------------------------------
        correct = torch.zeros((), device=device)
        preserve = torch.zeros((), device=device)
        identity = torch.zeros((), device=device)
        clean = outputs.get("clean_tokens")
        if clean is not None:
            if self.lambda_correct > 0:
                # repair ONLY the corrupted positions -- averaging over all 256 lets the
                # ~200 clean tokens dilute the "fix the broken one" objective
                m_r = mask_r.to(device).float() if has_mask else torch.ones_like(correct)
                m_t = mask_t.to(device).float() if has_mask else torch.ones_like(correct)
                err_r = F.l1_loss(outputs["corrected_rgb"].float(), clean["rgb"].float(),
                                  reduction="none").mean(-1)          # B x N
                err_t = F.l1_loss(outputs["corrected_tir"].float(), clean["tir"].float(),
                                  reduction="none").mean(-1)          # B x N
                correct = ((err_r * m_r).sum() / m_r.sum().clamp(min=1.0)
                           + (err_t * m_t).sum() / m_t.sum().clamp(min=1.0))
                preserve = ((err_r * (1 - m_r)).sum() / (1 - m_r).sum().clamp(min=1.0)
                            + (err_t * (1 - m_t)).sum() / (1 - m_t).sum().clamp(min=1.0))
            if self.lambda_identity > 0:
                keys = F.normalize(self.identity_proj(outputs["identity_tokens"].float()), dim=-1)
                with torch.no_grad():
                    target_dist = self._identity_dist(clean["rgb"].float(), keys)
                pred_dist = self._identity_dist(outputs["corrected_rgb"].float(), keys)
                identity = -(target_dist * torch.log(pred_dist.clamp(min=1e-8))).sum(-1).mean()

        parts.update({"cls": cls.detach(), "l1": l1.detach(), "giou": giou.detach(),
                      "detect": detect.detach(), "correct": correct.detach(),
                      "preserve": preserve.detach(), "identity": identity.detach()})

        total = (track
                 + self.lambda_detect * detect
                 + self.lambda_correct * correct
                 + self.lambda_identity * identity)
        parts["track"] = track.detach()
        parts["loss"] = total
        return parts
