"""RBGBackbone — the dual-branch U-Net denoiser at the core of RBG-Diff.

The network described in the paper, named throughout in the paper's terminology.

Architecture
------------
Two fully-parallel branches share the encoder/decoder *shape* but keep
independent weights and per-branch timestep conditioning:

  Reconstruction branch  R_theta  (code prefix ``recon_``)
      1-channel image input (the current sparse-view recon), a standard
      encoder / bottleneck / decoder, and the final image head.

  Residual branch        R_psi    (code prefix ``resid_``)
      the two residual signals (input_residual, prev_res_out) are stacked along
      the batch dim and pushed through a weight-shared encoder as a pure
      residual stream; a supervised residual head produces res_out.

  BSRF  (Band-Selective Residual Fusion, code prefix ``bsrf_``)
      at every decoder skip the two residual halves are fused with the
      reconstruction skip *per frequency band* (per-(level, channel)
      FrequencyBandModulation + 1x1 conv). A dedicated bottleneck BSRF merges
      the two residual streams before the residual bottleneck.

  Cross-branch fusion    (code prefix ``cross_``)
      after each decoder level the residual-branch feature is injected into the
      reconstruction branch via a 1x1 conv (one-directional, res -> recon).

Input layout:  x[:, 0:1] = input_residual, x[:, 1:2] = prev_res_out,
               x[:, 2:3] = image (current sparse recon).
forward(x, t, t_res=None) -> (recon_out, res_out).
"""

import torch
import torch.nn as nn

from networks.diffunet import (
    ResnetBlock, AttnBlock, Upsample, Downsample, Normalize,
    get_timestep_embedding, nonlinearity,
)
from networks.bsrf import FrequencyBandModulation


class RBGBackbone(nn.Module):
    """Dual-branch (reconstruction R_theta / residual R_psi) U-Net with BSRF."""

    def __init__(self, ch, ch_mult=(1, 2, 2, 2), num_res_blocks=2,
                 attn_resolutions=(16,), dropout=0.1, resamp_with_conv=True,
                 with_time_emb=True, resolution=256, fbm_k_list=(2, 4, 8)):
        super().__init__()
        self.ch = ch
        self.temb_ch = ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.with_time_emb = with_time_emb

        temb_ch = self.temb_ch
        in_ch_mult = (1,) + tuple(ch_mult)

        # =============================================================
        # Reconstruction branch R_theta  (recon_*)
        # =============================================================
        self.recon_temb = nn.Module()
        self.recon_temb.dense = nn.ModuleList([
            torch.nn.Linear(ch, temb_ch),
            torch.nn.Linear(temb_ch, temb_ch),
        ])
        self.recon_conv_in = torch.nn.Conv2d(1, ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        self.recon_down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.recon_down.append(down)

        self.recon_mid = nn.Module()
        self.recon_mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                             temb_channels=temb_ch, dropout=dropout)
        self.recon_mid.attn_1 = AttnBlock(block_in)
        self.recon_mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                             temb_channels=temb_ch, dropout=dropout)

        self.recon_up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                if i_block == num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in, out_channels=block_out,
                                         temb_channels=temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.recon_up.insert(0, up)

        self.recon_norm_out = Normalize(block_in)
        self.recon_conv_out = torch.nn.Conv2d(block_in, 1, kernel_size=3, stride=1, padding=1)

        # =============================================================
        # Residual branch R_psi  (resid_*)
        # =============================================================
        self.resid_temb = nn.Module()
        self.resid_temb.dense = nn.ModuleList([
            torch.nn.Linear(ch, temb_ch),
            torch.nn.Linear(temb_ch, temb_ch),
        ])
        # weight-shared 1-channel input conv (two residual signals batch-stacked)
        self.resid_conv_in = torch.nn.Conv2d(1, ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        self.resid_down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in_l = ch * in_ch_mult[i_level]
            block_out_l = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in_l, out_channels=block_out_l,
                                         temb_channels=temb_ch, dropout=dropout))
                block_in_l = block_out_l
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in_l))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in_l, resamp_with_conv)
                curr_res = curr_res // 2
            self.resid_down.append(down)

        block_in = ch * ch_mult[-1]
        self.resid_mid = nn.Module()
        self.resid_mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                             temb_channels=temb_ch, dropout=dropout)
        self.resid_mid.attn_1 = AttnBlock(block_in)
        self.resid_mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                             temb_channels=temb_ch, dropout=dropout)

        block_in = ch * ch_mult[-1]
        curr_res = resolution // (2 ** (self.num_resolutions - 1))
        self.resid_up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn_blks = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                if i_block == num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in, out_channels=block_out,
                                         temb_channels=temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn_blks.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn_blks
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.resid_up.insert(0, up)

        # residual output head (GroupNorm default eps=1e-5, matching the original head)
        res_out_ch = ch * ch_mult[0]
        self.resid_norm_out = nn.GroupNorm(32, res_out_ch)
        self.resid_conv_out = nn.Conv2d(res_out_ch, 1, kernel_size=3, stride=1, padding=1)

        # =============================================================
        # Cross-branch fusion (res -> recon, per decoder level)  (cross_*)
        # =============================================================
        self.cross_fuse = nn.ModuleList([
            nn.Conv2d(2 * (ch * ch_mult[i_level]), ch * ch_mult[i_level],
                      kernel_size=1, bias=True)
            for i_level in range(self.num_resolutions)
        ])

        # =============================================================
        # BSRF — Band-Selective Residual Fusion  (bsrf_*)
        # =============================================================
        # Decoder: per-(scale-level, channel) independent band modulation + 1x1 fuse.
        self._level_channels = {
            i: sorted({ch * in_ch_mult[i], ch * ch_mult[i]})
            for i in range(self.num_resolutions)
        }
        self.bsrf_fbm_in = nn.ModuleDict()
        self.bsrf_fbm_out = nn.ModuleDict()
        self.bsrf_fuse = nn.ModuleDict()
        for lvl, chans in self._level_channels.items():
            for c in chans:
                k = f'L{lvl}_C{c}'
                self.bsrf_fbm_in[k] = FrequencyBandModulation(c, k_list=fbm_k_list, lowfreq_att=True)
                self.bsrf_fbm_out[k] = FrequencyBandModulation(c, k_list=fbm_k_list, lowfreq_att=True)
                self.bsrf_fuse[k] = nn.Conv2d(3 * c, c, kernel_size=1, bias=True)

        # Bottleneck: single channel-shared band-modulation pair + 1x1 fuse.
        c_bn = ch * ch_mult[-1]
        self.bsrf_bn_fbm_in = FrequencyBandModulation(c_bn, k_list=fbm_k_list, lowfreq_att=True)
        self.bsrf_bn_fbm_out = FrequencyBandModulation(c_bn, k_list=fbm_k_list, lowfreq_att=True)
        self.bsrf_bn_fuse = nn.Conv2d(3 * c_bn, c_bn, kernel_size=1, bias=True)

    # ------------------------------------------------------------------
    # BSRF fusion helpers
    # ------------------------------------------------------------------
    def _bsrf_fuse_skip(self, h_in, h_out, recon, level):
        """Per-(level, channel) band-selective fusion at a decoder skip.

        h_in/h_out: the two residual-stream halves (B, C, H, W);
        recon: the channel-matched reconstruction skip used as band-gain
        conditioning; returns the fused (B, C, H, W) residual skip.
        """
        c = h_in.shape[1]
        k = f'L{level}_C{c}'
        fi = self.bsrf_fbm_in[k](h_in, att_feat=recon)
        fo = self.bsrf_fbm_out[k](h_out, att_feat=recon)
        return self.bsrf_fuse[k](torch.cat([fi, fo, recon], dim=1))

    def _bsrf_fuse_bottleneck(self, h_in_bn, h_out_bn, recon_bn):
        """Bottleneck band-selective merge of the two residual streams."""
        fi = self.bsrf_bn_fbm_in(h_in_bn, att_feat=recon_bn)
        fo = self.bsrf_bn_fbm_out(h_out_bn, att_feat=recon_bn)
        return self.bsrf_bn_fuse(torch.cat([fi, fo, recon_bn], dim=1))

    # ------------------------------------------------------------------
    def _forward_bodies(self, x, t, t_res=None):
        B = x.shape[0]
        x_in_1ch = x[:, 0:1]      # input_residual
        x_out_1ch = x[:, 1:2]     # prev_res_out
        x_deg = x[:, 2:3]         # image (current sparse recon)
        assert x_deg.shape[2] == x_deg.shape[3] == self.resolution

        if t is None:
            assert not self.with_time_emb
            t = torch.full((B,), 0, dtype=torch.long, device=x.device)
        if t_res is None:
            t_res = t

        # ---- reconstruction-branch time embedding ----
        temb_main = get_timestep_embedding(t, self.ch)
        temb_main = self.recon_temb.dense[0](temb_main)
        temb_main = nonlinearity(temb_main)
        temb_main = self.recon_temb.dense[1](temb_main)

        # ---- residual-branch time embedding ----
        temb_res = get_timestep_embedding(t_res, self.ch)
        temb_res = self.resid_temb.dense[0](temb_res)
        temb_res = nonlinearity(temb_res)
        temb_res = self.resid_temb.dense[1](temb_res)
        temb_res_2B = temb_res.repeat(2, 1)

        # ---- reconstruction encoder ----
        hs_main = [self.recon_conv_in(x_deg)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.recon_down[i_level].block[i_block](hs_main[-1], temb_main)
                if len(self.recon_down[i_level].attn) > 0:
                    h = self.recon_down[i_level].attn[i_block](h)
                hs_main.append(h)
            if i_level != self.num_resolutions - 1:
                hs_main.append(self.recon_down[i_level].downsample(hs_main[-1]))

        h_main = hs_main[-1]
        h_main = self.recon_mid.block_1(h_main, temb_main)
        h_main = self.recon_mid.attn_1(h_main)
        h_main = self.recon_mid.block_2(h_main, temb_main)

        # ---- residual encoder: two-stream batch-stacked, PURE residual ----
        x_stacked = torch.cat([x_in_1ch, x_out_1ch], dim=0)
        h_stacked = self.resid_conv_in(x_stacked)
        hs_res = []  # each entry is a 2B stacked skip
        for i_level in range(self.num_resolutions):
            hs_res.append(h_stacked)  # level-start skip
            for i_block in range(self.num_res_blocks):
                h_stacked = self.resid_down[i_level].block[i_block](h_stacked, temb_res_2B)
                if len(self.resid_down[i_level].attn) > 0:
                    h_stacked = self.resid_down[i_level].attn[i_block](h_stacked)
                hs_res.append(h_stacked)
            if i_level != self.num_resolutions - 1:
                h_stacked = self.resid_down[i_level].downsample(h_stacked)

        # ---- bottleneck stream merge via BSRF ----
        h_res = self._bsrf_fuse_bottleneck(h_stacked[:B], h_stacked[B:], h_main)
        h_res = self.resid_mid.block_1(h_res, temb_res)
        h_res = self.resid_mid.attn_1(h_res)
        h_res = self.resid_mid.block_2(h_res, temb_res)

        # ---- dual decoder; residual skip = BSRF fusion of (in, out, recon) ----
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                skip_main = hs_main.pop()
                skip_res = hs_res.pop()                 # (2B, C, H, W)
                fused_res = self._bsrf_fuse_skip(skip_res[:B], skip_res[B:], skip_main, level=i_level)

                h_main = self.recon_up[i_level].block[i_block](
                    torch.cat([h_main, skip_main], dim=1), temb_main)
                h_res = self.resid_up[i_level].block[i_block](
                    torch.cat([h_res, fused_res], dim=1), temb_res)
                if len(self.recon_up[i_level].attn) > 0:
                    h_main = self.recon_up[i_level].attn[i_block](h_main)
                if len(self.resid_up[i_level].attn) > 0:
                    h_res = self.resid_up[i_level].attn[i_block](h_res)

            # cross-branch fusion (res -> recon)
            h_main = self.cross_fuse[i_level](torch.cat([h_main, h_res], dim=1))

            if i_level != 0:
                h_main = self.recon_up[i_level].upsample(h_main)
                h_res = self.resid_up[i_level].upsample(h_res)

        h_main = self.recon_norm_out(h_main)
        h_main = nonlinearity(h_main)
        recon_out = self.recon_conv_out(h_main)
        return recon_out, h_res

    def forward(self, x, t, t_res=None):
        recon_out, h_res = self._forward_bodies(x, t, t_res)
        h_res = self.resid_norm_out(h_res)
        h_res = nonlinearity(h_res)
        res_out = self.resid_conv_out(h_res)
        return recon_out, res_out
