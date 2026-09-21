"""RBGDiff — Residual-Bootstrapping Generalized Diffusion for sparse-view CT.

The model casts generalized-diffusion sparse-view CT training as a bootstrapped
residual-error learning problem on top of an iterative SNR-biased (descending
view-count) reconstruction. It wraps a dual-branch U-Net denoiser
(``RBGBackbone``) in an EPCT-style chain rollout over a fixed descending view
schedule:

* the reconstruction branch R_theta predicts the clean CT from the redegraded
  input, and
* the residual branch R_psi estimates the residual error left by the previous
  step, bootstrapping its own previous estimate together with an input-anchored
  residual.

Band-Selective Residual Fusion (BSRF) fuses the two residual cues into R_theta.
The chain starts with a "null step" that reconstructs directly from the SVCT
input with both residual cues zero.

Radon / sparse-view data-generation ops are inherited from
``DiffusionSparseWrapper``.
"""

import torch

from wrappers.geometry import DiffusionSparseWrapper
from networks.backbone import RBGBackbone
from datasets.aapm import CTTools
from utilities.residual_norm import normalize_residual_in


class RBGDiff(DiffusionSparseWrapper):
    """Residual-branch-guided cold diffusion for sparse-view CT reconstruction."""

    # Fixed descending sparse-view schedule (dense -> sparse); step count = len().
    VIEW_LIST = [288, 234, 180, 126, 72, 54, 36, 18]

    def __init__(self, opt, **wrapper_kwargs):
        super().__init__(**wrapper_kwargs)
        self.opt = opt
        self.denoise_fn = RBGBackbone(ch=opt.unet_dim)
        self.view_list = list(self.VIEW_LIST)
        self.num_timesteps = len(self.view_list)
        self.cttool = CTTools()
        self.sigma = opt.err_cfg_sigma
        n_groups = len(self.denoise_fn.bsrf_fbm_in)
        print(f'[RBGDiff] dual-branch U-Net (R_theta / R_psi) with per-(level,channel) '
              f'BSRF ({n_groups} decoder band-modulation groups, lowfreq_att=True)')

    # ------------------------------------------------------------------
    # Data generation (radon ops inherited from DiffusionSparseWrapper)
    # ------------------------------------------------------------------
    def generate_sparse_and_gt_data(self, mu_ct, num_views):
        sparse_mu, gt_mu = self.generate_sparse_and_full_ct(mu_ct, num_views=num_views)
        return sparse_mu, gt_mu

    def forward(self, x, t, t_res=None):
        return self.denoise_fn(x, t, t_res=t_res)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_chain(self, x_input, t_start):
        """EPCT chain-rollout sampling loop.

        Residual timing: the residual branch's propagation channel
        (prev_res_out) is zero for the null step and the first loop iteration,
        then carries the previous step's res_out. The residual input channel is
        per-sample magnitude-normalized against the current recon (train/test
        symmetric). Returns (final_recon, direct_recon).
        """
        self.denoise_fn.eval()
        b = x_input.shape[0]
        v_input = self.view_list[t_start]

        t_id_list = list(range(t_start, -1, -1))

        # ---- null step ----
        step_null = torch.full((b,), t_start, dtype=torch.long, device=x_input.device)
        zeros = torch.zeros_like(x_input)
        null_input = torch.cat([zeros, zeros, x_input], dim=1)
        x_hat_0, _res_out_0_discarded = self.denoise_fn(null_input, step_null)
        direct_recon = x_hat_0

        step_input_lvl = torch.full((b,), t_start, dtype=torch.long, device=x_input.device)
        x_hat_prev = x_hat_0

        # prev_res_out starts at zeros (used through chain_step 1; real propagation from step 2).
        prev_res_out = torch.zeros_like(x_input)

        for t_id in t_id_list:
            step = torch.full((b,), t_id, dtype=torch.long, device=x_input.device)

            deg_prev, _ = self.generate_sparse_and_full_ct(x_hat_prev, num_views=v_input)
            residual_ch = torch.tanh((deg_prev - x_input) / self.sigma)

            image_ch, _ = self.generate_sparse_and_full_ct(
                x_hat_prev, num_views=self.view_list[t_id])

            residual_ch = normalize_residual_in(residual_ch, image_ch)

            x_in = torch.cat([residual_ch, prev_res_out, image_ch], dim=1)
            x_hat_prev, res_out_curr = self.denoise_fn(
                x_in, step, t_res=step_input_lvl,
            )

            prev_res_out = res_out_curr

        return x_hat_prev, direct_recon

    @torch.no_grad()
    def iterative_sample(self, x_deg, t_start, iterative_budget, refine_budget=2):
        return self.sample_chain(x_deg, t_start)

    @torch.no_grad()
    def sample(self, x, t=None):
        t_start = self.num_timesteps - 1 if t is None else t
        return self.sample_chain(x, t_start)
