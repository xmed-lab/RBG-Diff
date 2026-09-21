"""BSRF — Band-Selective Residual Fusion.

This module holds the frequency-band modulation primitive used by RBG-Diff's
Band-Selective Residual Fusion (BSRF). At every decoder skip (and at the
bottleneck) the reconstruction branch R_theta and the residual branch R_psi are
fused *per frequency band*: each feature is decomposed into a Laplacian-pyramid
of bands, each band is reweighted by a learned, spatially-varying gain that is
conditioned on the reconstruction (main) feature, and the bands are recombined.
`RBGBackbone` (networks/backbone.py) owns the per-(level, channel) instances of
`FrequencyBandModulation` and the 1x1 fusion convs; this file provides the band
modulator itself.

The band decomposition is adapted from FDConv (CVPR2025, "Frequency Dynamic
Convolution", https://github.com/Linwei-Chen/FDConv); spatial_group=1
(channel-broadcast band gain), k_list=(2, 4, 8), lowfreq_att=True (the lowest
band is also gated).

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_fft2freq(d1, d2, use_rfft=True):
    """Sorted-by-radius rfft2 frequency-coordinate indices + the (d1,d2,2) freq grid."""
    freq_h = torch.fft.fftfreq(d1)
    freq_w = torch.fft.rfftfreq(d2) if use_rfft else torch.fft.fftfreq(d2)
    freq_hw = torch.stack(torch.meshgrid(freq_h, freq_w, indexing='ij'), dim=-1)
    dist = torch.norm(freq_hw, dim=-1)
    _, indices = torch.sort(dist.view(-1))
    dd2 = (d2 // 2 + 1) if use_rfft else d2
    sorted_coords = torch.stack([indices // dd2, indices % dd2], dim=-1)
    return sorted_coords.permute(1, 0), freq_hw


class FrequencyBandModulation(nn.Module):
    """Decompose a feature into Laplacian-pyramid frequency bands, apply a learned,
    spatially-varying, att-conditioned gain per band, then recombine.

    in_channels == channels of the att_feat (== channels of the modulated feature,
    since in BSRF the conditioning feature (reconstruction skip) is channel-matched
    to the residual halves being fused).
    """

    def __init__(self, in_channels, k_list=(2, 4, 8), lowfreq_att=False,
                 act='sigmoid', spatial_group=1, spatial_kernel=3,
                 init='zero', max_size=(256, 256)):
        super().__init__()
        self.k_list = list(k_list)
        self.lowfreq_att = lowfreq_att
        self.act = act
        self.disabled = False  # ablation hook: identity passthrough when True
        if spatial_group > 64:
            spatial_group = in_channels
        self.spatial_group = spatial_group

        self.freq_weight_conv_list = nn.ModuleList()
        _n = len(self.k_list) + (1 if lowfreq_att else 0)
        for _ in range(_n):
            conv = nn.Conv2d(in_channels, self.spatial_group, kernel_size=spatial_kernel,
                             padding=spatial_kernel // 2, groups=self.spatial_group, bias=True)
            if init == 'zero':
                nn.init.normal_(conv.weight, std=1e-6)
                if conv.bias is not None:
                    conv.bias.data.zero_()
            self.freq_weight_conv_list.append(conv)

        self.register_buffer('cached_masks',
                             self._precompute_masks(max_size, self.k_list), persistent=False)

    def _precompute_masks(self, max_size, k_list):
        max_h, max_w = max_size
        _, freq_indices = get_fft2freq(d1=max_h, d2=max_w, use_rfft=True)
        freq_mag = freq_indices.abs().max(dim=-1, keepdims=False)[0]  # (max_h, max_w//2+1)
        masks = [(freq_mag < 0.5 / freq + 1e-8) for freq in k_list]
        return torch.stack(masks, dim=0).unsqueeze(1)  # (num_masks, 1, H, W//2+1)

    def sp_act(self, w):
        if self.act == 'sigmoid':
            return w.sigmoid() * 2
        if self.act == 'tanh':
            return 1 + w.tanh()
        raise NotImplementedError

    def forward(self, x, att_feat=None):
        if att_feat is None:
            att_feat = x
        if self.disabled:
            return x
        x = x.to(torch.float32)
        pre_x = x.clone()
        b, _, h, w = x.shape
        x_fft = torch.fft.rfft2(x, norm='ortho')
        freq_h, freq_w = h, w // 2 + 1
        current_masks = F.interpolate(self.cached_masks.float(), size=(freq_h, freq_w), mode='nearest')

        x_list = []
        for idx, _freq in enumerate(self.k_list):
            mask = current_masks[idx]
            low_part = torch.fft.irfft2(x_fft * mask, s=(h, w), norm='ortho')
            high_part = pre_x - low_part
            pre_x = low_part
            fw = self.sp_act(self.freq_weight_conv_list[idx](att_feat))
            tmp = fw.reshape(b, self.spatial_group, -1, h, w) * \
                high_part.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))

        if self.lowfreq_att:
            fw = self.sp_act(self.freq_weight_conv_list[len(self.k_list)](att_feat))
            tmp = fw.reshape(b, self.spatial_group, -1, h, w) * \
                pre_x.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))
        else:
            x_list.append(pre_x)
        return sum(x_list)
