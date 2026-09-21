"""Per-sample residual-branch normalization helpers.

Scheme
------
For each batch element B:
  s_in  = torch.quantile(|residual_ch|.flatten(), 0.95)
  s_gt  = torch.quantile(|image_ch  |.flatten(), 0.95)
  s_out = torch.quantile(|target_res|.flatten(), 0.95)

  residual_ch_norm = (residual_ch / max(s_in,  eps)) * s_gt
  target_res_norm  = (target_res  / max(s_out, eps)) * s_gt

`s_gt = ||image_ch||_p95` is computed from a quantity available at BOTH train
and inference, ensuring zero train/test asymmetry on the input pathway.
`s_out` is training-only (target_res requires gt_mu) but never needed at
inference because res_out is supervision-only — the reconstruction path consumes
only main_out from the (main, res) tuple.

No clipping. Null step (residual_ch ≡ zeros) passes through cleanly: the
formula yields zeros via the eps fallback.
"""

import torch


def _per_sample_p95(x, eps=1e-6):
    """Per-sample 95th-percentile of |x|, returns (B,)."""
    abs_x = x.detach().abs().reshape(x.shape[0], -1)
    s = torch.quantile(abs_x, 0.95, dim=1)
    return s.clamp_min(eps)


def normalize_residual_in(residual_ch, image_ch, eps=1e-6):
    """Scale residual_ch into the s_gt magnitude reference.

    residual_ch: (B, 1, H, W) — tanh-bounded consistency error
    image_ch:    (B, 1, H, W) — sparse-view recon at the current step
    Returns:     (B, 1, H, W) — normalized residual ready for concat as ch0.
    """
    s_in = _per_sample_p95(residual_ch, eps)            # (B,)
    s_gt = _per_sample_p95(image_ch,    eps)            # (B,)
    scale = (s_gt / s_in).view(-1, 1, 1, 1)              # (B,1,1,1)
    return residual_ch * scale


def normalize_residual_target(target_res, image_ch, eps=1e-6):
    """Scale target_res into the s_gt magnitude reference (training only).

    target_res: (B, 1, H, W) — tanh-bounded image-space recon error
    image_ch:   (B, 1, H, W) — same image_ch used for the EPCT step input
    Returns:    (B, 1, H, W) — normalized supervision target for res_loss.
    """
    s_out = _per_sample_p95(target_res, eps)            # (B,)
    s_gt  = _per_sample_p95(image_ch,   eps)            # (B,)
    scale = (s_gt / s_out).view(-1, 1, 1, 1)
    return target_res * scale
