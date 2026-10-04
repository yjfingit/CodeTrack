"""Center-based tracking head.

Ported from OSTrack (Apache-2.0), https://github.com/botaoye/OSTrack
``lib/models/layers/head.py``.

The only change is device handling: the reference implementation hard-codes
``.cuda()`` for the coordinate grids, which is replaced by registered buffers so the
head works on any device.  Keeping the module byte-compatible otherwise means the
public OSTrack checkpoint's head weights load directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, freeze_bn=False):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class CenterPredictor(nn.Module):
    """Center heatmap + size regression + sub-cell offset, as in OSTrack."""

    def __init__(self, inplanes=768, channel=256, feat_sz=16, stride=16, freeze_bn=False):
        super().__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride

        # corner / centre branch
        self.conv1_ctr = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_ctr = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_ctr = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_ctr = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_ctr = nn.Conv2d(channel // 8, 1, kernel_size=1)

        # size regression
        self.conv1_offset = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_offset = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_offset = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_offset = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_offset = nn.Conv2d(channel // 8, 2, kernel_size=1)

        self.conv1_size = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_size = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_size = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_size = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_size = nn.Conv2d(channel // 8, 2, kernel_size=1)

        with torch.no_grad():
            indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            coord_x = indice.repeat((self.feat_sz, 1)).view(-1).float()
            coord_y = indice.repeat((1, self.feat_sz)).view(-1).float()
        self.register_buffer("coord_x", coord_x)
        self.register_buffer("coord_y", coord_y)

        # Optional inference-time score-map window (OSTrack applies a Hann window with a
        # `window_influence` before the argmax; this head does not).  Registered as a buffer so it
        # follows the module's device, and gated so the default path is bit-identical: the window
        # changes which cell wins the argmax, so it is an inference-protocol choice, not a
        # numerical detail.  See `eval.score_window` and docs/results.md 6.29.
        self.score_window = "none"
        self.window_influence = 0.5
        window = torch.outer(torch.hann_window(self.feat_sz, periodic=False),
                             torch.hann_window(self.feat_sz, periodic=False))
        self.register_buffer("hann_window", window.view(1, 1, self.feat_sz, self.feat_sz))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------ forward
    def forward(self, x, gt_score_map=None):
        score_map_ctr, size_map, offset_map = self.get_score_map(x)
        bbox = self.cal_bbox(gt_score_map.unsqueeze(1) if gt_score_map is not None
                             else score_map_ctr, size_map, offset_map)
        return score_map_ctr, bbox, size_map, offset_map

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        if self.score_window == "hann":
            # The centre head picks the global argmax of the score map, so a spurious high response
            # far from the previous position wins outright.  A Hann window biases the choice
            # towards the middle of the window (where the target was when the crop was taken), which
            # is what OSTrack's `window_influence` does.  Applied here rather than in `forward` so
            # the *returned* `score_map_ctr` keeps its semantics.
            score_map_ctr = (score_map_ctr * (1.0 - self.window_influence)
                             + self.hann_window.to(score_map_ctr.dtype)
                             * self.window_influence)
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)      # cx, cy, w, h in [0, 1]

        if return_score:
            return bbox, max_score
        return bbox

    def get_score_map(self, x):
        def _sigmoid(t):
            return torch.clamp(t.sigmoid_(), min=1e-4, max=1 - 1e-4)

        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map_ctr = self.conv5_ctr(x_ctr4)

        x_off1 = self.conv1_offset(x)
        x_off2 = self.conv2_offset(x_off1)
        x_off3 = self.conv3_offset(x_off2)
        x_off4 = self.conv4_offset(x_off3)
        score_map_offset = self.conv5_offset(x_off4)

        x_size1 = self.conv1_size(x)
        x_size2 = self.conv2_size(x_size1)
        x_size3 = self.conv3_size(x_size2)
        x_size4 = self.conv4_size(x_size3)
        score_map_size = self.conv5_size(x_size4)
        return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset
