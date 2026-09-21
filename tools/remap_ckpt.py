"""Checkpoint key mapping for RBG-Diff.

Rewrites the second-level module names under ``denoise_fn.`` to RBG-Diff's
naming. Weights are unchanged; only state_dict keys are renamed. Every rename is
a second-level token substitution (``denoise_fn.<token>.<...>``).

Usage:
    python tools/remap_ckpt.py <old_ckpt> <new_ckpt>
"""

import sys
import torch

# old second-level token -> new (paper-named) token, under the 'denoise_fn.' root.
TOKEN_MAP = {
    # reconstruction branch R_theta
    'temb':               'recon_temb',
    'conv_in':            'recon_conv_in',
    'down':               'recon_down',
    'mid':                'recon_mid',
    'up':                 'recon_up',
    'norm_out':           'recon_norm_out',
    'conv_out':           'recon_conv_out',
    # residual branch R_psi
    'res_temb':           'resid_temb',
    'res_conv_in_shared': 'resid_conv_in',
    'res_down':           'resid_down',
    'res_mid':            'resid_mid',
    'res_up':             'resid_up',
    'norm_out_res':       'resid_norm_out',
    'res_out_conv':       'resid_conv_out',
    # cross-branch fusion (res -> recon)
    'fuse_conv':          'cross_fuse',
    # BSRF — Band-Selective Residual Fusion (decoder, per-(level,channel))
    'fbm_in':             'bsrf_fbm_in',
    'fbm_out':            'bsrf_fbm_out',
    'skip_fuse':          'bsrf_fuse',
    # BSRF bottleneck
    'fbm_in_bn':          'bsrf_bn_fbm_in',
    'fbm_out_bn':         'bsrf_bn_fbm_out',
    'skip_fuse_bn':       'bsrf_bn_fuse',
}

ROOT = 'denoise_fn.'


def remap_key(key):
    if not key.startswith(ROOT):
        return key
    rest = key[len(ROOT):]
    token, _, tail = rest.partition('.')
    if token not in TOKEN_MAP:
        raise KeyError(f'Unmapped second-level token {token!r} in key {key!r}')
    new_token = TOKEN_MAP[token]
    return ROOT + new_token + ('.' + tail if tail else '')


def remap_state_dict(sd):
    new_sd = {}
    for k, v in sd.items():
        nk = remap_key(k)
        if nk in new_sd:
            raise KeyError(f'Key collision: {k!r} and another map to {nk!r}')
        new_sd[nk] = v
    assert len(new_sd) == len(sd), 'key count changed during remap'
    return new_sd


def main():
    if len(sys.argv) != 3:
        print('usage: python tools/remap_ckpt.py <old_ckpt> <new_ckpt>')
        sys.exit(1)
    old_path, new_path = sys.argv[1], sys.argv[2]
    ckpt = torch.load(old_path, map_location='cpu')
    assert 'net_param' in ckpt, 'expected {net_param, epoch} checkpoint'
    new_ckpt = {'net_param': remap_state_dict(ckpt['net_param'])}
    if 'epoch' in ckpt:
        new_ckpt['epoch'] = ckpt['epoch']
    torch.save(new_ckpt, new_path)
    print(f'remapped {len(ckpt["net_param"])} keys -> {new_path}; epoch={new_ckpt.get("epoch")}')


if __name__ == '__main__':
    main()
