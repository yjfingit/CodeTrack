"""Corner-based tracking head.

Ported from OSTrack (Apache-2.0), https://github.com/botaoye/OSTrack
``lib/models/layers/head.py``.  Device handling adapted (coordinate grids are
buffers instead of hard-coded ``.cuda()`` tensors).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .center import conv


class CornerPredictor(nn.Module):
    """Top-left / bottom-right corner heatmaps with soft-argmax decoding."""

    def __init__(self, inplanes=768, channel=256, feat_sz=16, stride=16, freeze_bn=False):
        super().__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride

        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        with torch.no_grad():
            indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            coord_x = indice.repeat((self.feat_sz, 1)).view(-1).float()
            coord_y = indice.repeat((1, self.feat_sz)).view(-1).float()
        self.register_buffer("coord_x", coord_x)
        self.register_buffer("coord_y", coord_y)

    def forward(self, x, return_dist=False, softmax=True):
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            cx_tl, cy_tl, p_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            cx_br, cy_br, p_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((cx_tl, cy_tl, cx_br, cy_br), dim=1) / self.img_sz, p_tl, p_br
        cx_tl, cy_tl = self.soft_argmax(score_map_tl)
        cx_br, cy_br = self.soft_argmax(score_map_br)
        return torch.stack((cx_tl, cy_tl, cx_br, cy_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        x_tl = self.conv5_tl(self.conv4_tl(self.conv3_tl(self.conv2_tl(self.conv1_tl(x)))))
        x_br = self.conv5_br(self.conv4_br(self.conv3_br(self.conv2_br(self.conv1_br(x)))))
        return x_tl, x_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            return exp_x, exp_y, (prob_vec if softmax else score_vec)
        return exp_x, exp_y
