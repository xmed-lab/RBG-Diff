"""RBGDiffTrainer — the persistence-aware EPCT chain-rollout trainer for RBG-Diff.

A single, paper-named trainer on top of the generic ``BasicTrainer`` base. It
runs the EPCT chain rollout against a frozen EMA teacher and applies the
persistence-aware guidance losses (L_pa spatial + L_fpa spectral). It provides:
    * the persistence-aware training loop,
    * the EMA / fit / val machinery, and
    * (from BasicTrainer) optimizer, scheduler, and checkpoint I/O.

Loss naming (paper <-> code map)
--------------------------------
    L_pa  (persistence-aware SPATIAL loss)   = ``l_pa``   (was ``loss_res``):
        per-pixel persistence-rank-weighted focal MSE on the residual head,
        with plain MSE fallback. Weighted by ``res_loss_weight`` (paper lambda_pa).
    L_fpa (persistence-aware SPECTRAL loss)  = ``l_fpa``  (was ``l_freqpa``):
        per-frequency chain-persistence-weighted focal on |FFT(res_out)-FFT(target)|^2.
        Weighted by ``freqpa_weight`` (paper lambda_fpa), enabled by ``use_freq_pa``.
    FFL   (focal frequency loss, ``use_ffl``): OFF in the final recipe — the flag
        and code path are kept for the ablation API but unused when use_ffl=False.

Flag map (CLI kept as-is; documented paper term):
    res_loss_weight   -> lambda_pa       (weight on L_pa)
    gamma_max         -> gamma_pa^max    (focal exponent for the spatial rank weight)
    freqpa_weight     -> lambda_fpa      (weight on L_fpa)
    freqpa_gamma_max  -> gamma_fpa^max   (focal exponent for the spectral rank weight)
    gamma_ramp_iters  -> cosine ramp horizon for both focal exponents
    epct_ramp_iters   -> cosine ramp horizon for the EPCT (chain-rollout) coefficient
    err_cfg_sigma     -> sigma           (tanh residual scale)

Mechanism (linear_late + percentile-rank stacking):
  For each iter t_id in the EMA chain rollout, compute err_k = |x_hat_ema_k - gt_mu|.
  Convert err_k to continuous per-pixel percentile rank ranks_k in [0, 1].
  Accumulate persistence_weighted += step_weight * ranks_k (step_weight linear_late),
  persistence_score = persistence_weighted / total_step_weight in [0, 1].
  weight = persistence_score ** gamma, per-sample normalized; L_pa = (weight * sq_err).mean().
  gamma cosine-ramps 0 -> gamma_max over gamma_ramp_iters iters.
"""

import os
import math
import copy
import wandb
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append('..')

from trainers.basic_trainer import BasicTrainer
from datasets.aapm import AAPMMyoDataset
from utilities.residual_norm import (
    normalize_residual_in,
    normalize_residual_target,
)


class EMA:
    """Exponential Moving Average — borrowed unchanged from CvG-Diff."""
    def __init__(self, beta):
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class RBGDiffTrainer(BasicTrainer):
    """Persistence-aware EPCT trainer for RBG-Diff. Single-GPU.

    EPCT (Exact-Path Chain Training): the EMA net is rolled out through the exact
    sequential inference chain to build the training input, and the persistence of
    the residual error along the chain drives the focal weighting of both the
    spatial (L_pa) and spectral (L_fpa) residual-head losses.
    """

    def __init__(self, opt=None, net=None, loss_type='l2'):
        super().__init__()
        self.opt = opt
        self.net = net
        self.sigma = opt.err_cfg_sigma

        dataset_name = opt.dataset_name.lower()
        if dataset_name == 'aapm':
            self.train_dataset = AAPMMyoDataset(opt.dataset_path, mode='train', dataset_shape=opt.dataset_shape)
            self.val_dataset = AAPMMyoDataset(opt.dataset_path, mode='val', dataset_shape=opt.dataset_shape)
        else:
            raise NotImplementedError(f'Dataset {dataset_name} not implemented, try aapm.')

        self.checkpoint_path = os.path.join(opt.checkpoint_root, opt.checkpoint_dir)

        if opt.use_wandb:
            self.wandb_init(opt, key=getattr(opt, 'wandb_api_key', None))

        self.best_val_loss = np.inf
        self.update_ema_iter = opt.update_ema_iter
        self.start_ema_iter = opt.start_ema_iter
        self.ema_net = copy.deepcopy(net)
        for param in self.ema_net.parameters():
            param.requires_grad = False
        self.ema = EMA(opt.ema_decay)
        self.criterion = self.get_pixel_criterion(loss_type)
        self.itlog_intv = opt.log_interval

        # ---- EPCT + persistence-aware loss hyperparameters ----
        self.epct_ramp_iters = getattr(opt, 'epct_ramp_iters', self.start_ema_iter)
        self.res_loss_weight = float(getattr(opt, 'res_loss_weight', 0.1))   # lambda_pa

        # L_pa (spatial) focal schedule
        self.gamma_max = float(getattr(opt, 'gamma_max', 8.0))               # gamma_pa^max
        self.gamma_ramp_iters = int(getattr(opt, 'gamma_ramp_iters', 48000))
        self.step_weight_schedule = str(getattr(opt, 'step_weight_schedule', 'linear_late'))
        assert self.step_weight_schedule in ('linear_late',), (
            f'unsupported step_weight_schedule: {self.step_weight_schedule}')

        # Focal Frequency Loss (FFL) — OFF in the final recipe; kept for the ablation API.
        self.use_ffl = bool(getattr(opt, 'use_ffl', False))
        self.ffl_weight = float(getattr(opt, 'ffl_weight', 1.0))
        self.ffl_alpha = float(getattr(opt, 'ffl_alpha', 1.0))

        # L_fpa (spectral) — chain-persistent per-frequency focal on the residual head.
        self.use_freq_pa = bool(getattr(opt, 'use_freq_pa', False))
        self.freqpa_gamma_max = float(getattr(opt, 'freqpa_gamma_max', 8.0))  # gamma_fpa^max
        self.freqpa_weight = float(getattr(opt, 'freqpa_weight', 1.0))        # lambda_fpa

        print(f'[RBGDiffTrainer] epct_ramp_iters={self.epct_ramp_iters} '
              f'res_loss_weight(lambda_pa)={self.res_loss_weight} '
              f'gamma_max={self.gamma_max} '
              f'gamma_ramp_iters={self.gamma_ramp_iters} '
              f'step_weight_schedule={self.step_weight_schedule} '
              f'use_ffl={self.use_ffl} ffl_weight={self.ffl_weight} ffl_alpha={self.ffl_alpha} '
              f'use_freq_pa={self.use_freq_pa} freqpa_gamma_max={self.freqpa_gamma_max} '
              f'freqpa_weight(lambda_fpa)={self.freqpa_weight}')

    # ------------------------------------------------------------------
    # EMA / dataset / checkpoint helpers
    # ------------------------------------------------------------------
    @staticmethod
    def wandb_init(opt, key=None):
        if key is None:
            print('WANDB key not provided, attempting anonymous login...')
        else:
            wandb.login(key=key)
        wandb_root = opt.wandb_root
        wandb_dir = opt.wandb_dir
        wandb_path = os.path.join(wandb_root, wandb_dir)
        if not os.path.exists(wandb_path):
            os.makedirs(wandb_path)
        wandb.init(project=opt.wandb_project, entity=opt.wandb_entity, name=opt.run_name, config=opt)

    def save_opt(self, optimizer=None, scheduler=None, opt_name=''):
        checkpoint_path = os.path.join(self.opt.checkpoint_root, self.opt.checkpoint_dir)
        optimizer_param = optimizer.state_dict() if optimizer is not None else self.optimizer.state_dict()
        if scheduler is not None:
            opt_check = {'optimizer': optimizer_param, 'scheduler': scheduler.state_dict(),
                         'epoch': self.epoch, 'iter': self.iter}
        else:
            opt_check = {'optimizer': optimizer_param, 'epoch': self.epoch, 'iter': self.iter}
        self.save_checkpoint(opt_check, checkpoint_path, self.opt.checkpoint_dir + '-opt-' + opt_name, 'latest')

    def reset_parameters(self):
        self.ema_net.load_state_dict(self.net.state_dict())

    def step_ema(self, n_iter):
        if n_iter < self.start_ema_iter:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_net, self.net)

    def generate_sparse_and_gt_data(self, mu_ct, num_views):
        sparse_mu, gt_mu = self.net.generate_sparse_and_full_ct(mu_ct, num_views=num_views)
        return sparse_mu, gt_mu

    # ------------------------------------------------------------------
    # Persistence-aware loss helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _spectral_sq_err(pred, target):
        """|FFT(pred) - FFT(target)|^2 over the full 2D frequency plane, (B,1,H,W) real.
        Differentiable in pred (used for L_fpa) and reusable detached (rollout E_k)."""
        diff = torch.fft.fft2(pred, norm='ortho') - torch.fft.fft2(target, norm='ortho')
        return diff.real ** 2 + diff.imag ** 2

    @staticmethod
    def focal_frequency_loss(pred, target, alpha=1.0):
        """Focal Frequency Loss (Jiang et al., ICCV2021). Unused in the final recipe."""
        fp = torch.fft.fft2(pred, norm='ortho')
        ft = torch.fft.fft2(target, norm='ortho')
        diff = fp - ft
        d = diff.real ** 2 + diff.imag ** 2                  # |diff|^2  (B,1,H,W)
        with torch.no_grad():
            w = d.clamp_min(0.0).pow(alpha / 2.0)            # |diff|^alpha
            wmax = w.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
            w = (w / wmax)
        return (w * d).mean()

    @staticmethod
    def _per_pixel_percentile_rank(err):
        """err: (B, 1, H, W) -> ranks in [0, 1] per-sample (argsort/argsort)."""
        b = err.shape[0]
        flat = err.reshape(b, -1)
        n = flat.shape[-1]
        ranks = flat.argsort(dim=-1).argsort(dim=-1).float() / max(n - 1, 1)
        return ranks.reshape_as(err)

    # ------------------------------------------------------------------
    # Training loop (persistence-aware EPCT chain rollout)
    # ------------------------------------------------------------------
    def train(self):
        self.iter_log_flag = False
        losses, rmses, psnrs, ssims = [], [], [], []

        self.net.train()
        self.ema_net.train()
        pbar = tqdm.tqdm(self.train_loader, ncols=60) if self.opt.use_tqdm else self.train_loader

        # iter-level debug stats (sampled, not per-iter, to avoid log spam)
        last_persistence_stats = {'std': 0.0, 'max': 0.0, 'min': 0.0, 'fallback': True}

        for i, data in enumerate(pbar):
            mu_ct = data.to('cuda')
            b = mu_ct.shape[0]

            t_single = torch.randint(0, self.net.num_timesteps, (1,), device=mu_ct.device).long()
            t = t_single.repeat((b,))
            num_views = self.net.view_list[t_single.item()]
            sparse_mu, gt_mu = self.generate_sparse_and_gt_data(mu_ct, num_views=num_views)
            zeros = torch.zeros_like(sparse_mu)
            null_input = torch.cat([zeros, zeros, sparse_mu], dim=1)

            progress = min(1.0, self.iter / max(1, self.epct_ramp_iters))
            epct_ramp = 0.5 * (1.0 - math.cos(math.pi * progress))

            # gamma cosine-ramp 0 -> gamma_max over gamma_ramp_iters
            gamma_progress = min(1.0, self.iter / max(1, self.gamma_ramp_iters))
            gamma = self.gamma_max * 0.5 * (1.0 - math.cos(math.pi * gamma_progress))

            self.optimizer.zero_grad()

            # ---- null step on LIVE net (no res supervision; all_sup=False) ----
            out_null, _ = self.net(null_input, t, t_res=t)
            mse_null_val = self.criterion(out_null, gt_mu)
            total_null = mse_null_val
            total_null.backward()
            out_null = out_null.detach()

            chain_step = 1
            persistence_score = None  # (B, 1, H, W) in [0, 1] if computed
            persistence_fallback = True
            freq_persistence_score = None  # (B, 1, H, W) over freq bins, in [0, 1]

            if t_single.item() > 0:
                t_nxt = torch.randint(0, t_single.item() + 1, (1,), device=mu_ct.device).long()
                t_input = t_nxt.repeat((b,))
                vt_idx = t_nxt.item()

                with torch.no_grad():
                    x_hat_ema, _ = self.ema_net(null_input, t, t_res=t)
                    x_hat_ema = x_hat_ema.detach()

                    # Resolution E: chain step 1 sees zeros for channel-1.
                    prev_res_out_ema = torch.zeros_like(x_hat_ema)

                    # Persistence accumulator
                    chain_step_total = t_single.item() - t_nxt.item()
                    persistence_weighted = torch.zeros_like(gt_mu)
                    freq_persist_weighted = torch.zeros_like(gt_mu)  # L_fpa (per-frequency)
                    total_step_weight = 0.0
                    chain_step_idx = 0

                    for t_id in range(t_single.item(), t_nxt.item(), -1):
                        chain_step_idx += 1

                        x_hat_in = x_hat_ema  # input recon to this step (defines the head's target_k)
                        hat_input, _ = self.generate_sparse_and_gt_data(x_hat_ema, num_views=num_views)
                        residual_ch = torch.tanh((hat_input - sparse_mu) / self.sigma)
                        num_views_step = self.net.view_list[t_id]
                        image_ch, _ = self.generate_sparse_and_gt_data(x_hat_ema, num_views=num_views_step)
                        residual_ch = normalize_residual_in(residual_ch, image_ch)
                        t_id_t = torch.full((b,), t_id, dtype=torch.long, device=mu_ct.device)

                        ema_in = torch.cat([residual_ch, prev_res_out_ema, image_ch], dim=1)
                        x_hat_ema, res_out_ema = self.ema_net(ema_in, t_id_t, t_res=t)
                        x_hat_ema = x_hat_ema.detach()
                        res_out_ema = res_out_ema.detach()
                        prev_res_out_ema = res_out_ema
                        chain_step += 1

                        # Accumulate spatial persistence: weight by linear_late step_weight
                        err_k = (x_hat_ema - gt_mu).abs()
                        ranks_k = self._per_pixel_percentile_rank(err_k)
                        step_weight = chain_step_idx / max(chain_step_total, 1)
                        persistence_weighted += step_weight * ranks_k
                        total_step_weight += step_weight

                        # L_fpa: per-frequency persistence of the RESIDUAL-HEAD spectral error.
                        if self.use_freq_pa:
                            target_k = normalize_residual_target(
                                torch.tanh((x_hat_in - gt_mu) / self.sigma), image_ch)
                            ek = self._spectral_sq_err(res_out_ema, target_k)   # (B,1,H,W) freq
                            frank_k = self._per_pixel_percentile_rank(ek)        # rank over freq bins
                            freq_persist_weighted += step_weight * frank_k

                    if total_step_weight > 0.0 and chain_step_total > 0:
                        persistence_score = persistence_weighted / total_step_weight
                        persistence_fallback = False
                        if self.use_freq_pa:
                            freq_persistence_score = freq_persist_weighted / total_step_weight

                x_hat_best = x_hat_ema

                hat_input, _ = self.generate_sparse_and_gt_data(x_hat_best, num_views=num_views)
                residual_ch = torch.tanh((hat_input - sparse_mu) / self.sigma)
                num_views_nxt = self.net.view_list[vt_idx]
                image_ch, _ = self.generate_sparse_and_gt_data(x_hat_best, num_views=num_views_nxt)
                residual_ch = normalize_residual_in(residual_ch, image_ch)

                if chain_step == 1:
                    # Loop body had 0 iterations (t_nxt == t_single); EPCT is "first non-null step".
                    epct_prev_res = torch.zeros_like(residual_ch)
                else:
                    epct_prev_res = prev_res_out_ema

                epct_input = torch.cat([residual_ch, epct_prev_res, image_ch], dim=1)

            else:
                # t_single == 0: no EMA chain (and no persistence accumulator).
                t_input = t
                vt_idx = 0
                with torch.no_grad():
                    x_hat_ema_full = self.ema_net(null_input, t, t_res=t)
                    x_hat_ema = x_hat_ema_full[0].detach()
                x_hat_best = x_hat_ema
                image_ch = x_hat_best
                epct_prev_res = torch.zeros_like(zeros)
                epct_input = torch.cat([zeros, epct_prev_res, image_ch], dim=1)

            out_epct, res_out = self.net(epct_input, t_input, t_res=t)
            mse_epct_val = self.criterion(out_epct, gt_mu)

            target_res = torch.tanh((x_hat_best - gt_mu) / self.sigma)
            target_res_norm = normalize_residual_target(target_res, image_ch)

            # ---- L_pa: persistence-aware SPATIAL residual loss ----
            sq_err = (res_out - target_res_norm).pow(2)
            if persistence_score is not None and gamma > 0.0:
                # Persistence-weighted focal MSE on residual head.
                weight = persistence_score.pow(gamma)
                weight = weight / weight.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
                l_pa = (weight * sq_err).mean()

                # Track diagnostics (for iter logging)
                with torch.no_grad():
                    last_persistence_stats = {
                        'std': weight.std().item(),
                        'max': weight.max().item(),
                        'min': weight.min().item(),
                        'fallback': False,
                    }
            else:
                # Fallback: plain MSE.
                l_pa = sq_err.mean()
                if persistence_fallback or gamma == 0.0:
                    last_persistence_stats = {'std': 0.0, 'max': 1.0, 'min': 1.0, 'fallback': True}

            # ---- optional FFL on res_out (OFF in final recipe) ----
            # Ramps with the SAME cosine schedule as the persistence gamma.
            if self.use_ffl:
                ffl_ramp = 0.5 * (1.0 - math.cos(math.pi * gamma_progress))
                ffl_coef = self.ffl_weight * ffl_ramp
                l_ffl = self.focal_frequency_loss(res_out, target_res_norm, alpha=self.ffl_alpha)
                last_ffl_val = l_ffl.item()
                last_ffl_coef = ffl_coef
                res_branch_loss = self.res_loss_weight * l_pa + ffl_coef * l_ffl
            else:
                last_ffl_val = 0.0
                last_ffl_coef = 0.0
                res_branch_loss = self.res_loss_weight * l_pa

            # ---- L_fpa: persistence-aware SPECTRAL residual loss (chain-persistent focal) ----
            # focal weight = freq_persistence^freqpa_gamma (gamma ramps 0->max like pa/ffl);
            # overall contribution ramps with the same cosine schedule.
            if self.use_freq_pa:
                fp_ramp = 0.5 * (1.0 - math.cos(math.pi * gamma_progress))
                freqpa_gamma = self.freqpa_gamma_max * fp_ramp
                freqpa_coef = self.freqpa_weight * fp_ramp
                sq_spec = self._spectral_sq_err(res_out, target_res_norm)   # (B,1,H,W), grad in res_out
                if freq_persistence_score is not None and freqpa_gamma > 0.0:
                    fw = freq_persistence_score.pow(freqpa_gamma)
                    fw = fw / fw.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
                    l_fpa = (fw * sq_spec).mean()
                else:
                    l_fpa = sq_spec.mean()
                last_freqpa_val = l_fpa.item()
                last_freqpa_coef = freqpa_coef
                res_branch_loss = res_branch_loss + freqpa_coef * l_fpa
            else:
                last_freqpa_val = 0.0
                last_freqpa_coef = 0.0

            (epct_ramp * (mse_epct_val + res_branch_loss)).backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()

            losses.append((mse_null_val + epct_ramp * (mse_epct_val + self.res_loss_weight * l_pa)).item())
            rmse, psnr, ssim = self.get_metrics_by_window(out_null, gt_mu)
            rmses.append(rmse); psnrs.append(psnr); ssims.append(ssim)

            if self.iter != 0 and self.iter % self.itlog_intv == 0:
                if self.opt.use_wandb:
                    self.wandb_logger('train/iter', **{
                        'loss': np.mean(losses[-self.itlog_intv:]),
                        'psnr': np.mean(psnrs[-self.itlog_intv:]),
                        'epct_ramp': epct_ramp,
                        'gamma': gamma,
                        'l_pa_weight_std': last_persistence_stats['std'],
                        'l_pa_weight_max': last_persistence_stats['max'],
                    })

            self.iter += 1
            if self.iter % self.update_ema_iter == 0:
                self.step_ema(self.iter)

            if self.iter % 100 == 0:
                t_nxt_val = t_nxt.item() if t_single.item() > 0 else 0
                print(f'[RBGDiff iter {self.iter}] ramp={epct_ramp:.4f} '
                      f'gamma={gamma:.3f} '
                      f'mse_null={mse_null_val.item():.4f} '
                      f'mse_epct={mse_epct_val.item():.4f} '
                      f'L_pa={l_pa.item():.4f} '
                      f'lambda_pa={self.res_loss_weight:.2f} '
                      f'chain_step={chain_step} '
                      f't_single={t_single.item()} t_nxt={t_nxt_val} '
                      f'L_pa_w[std={last_persistence_stats["std"]:.3f} '
                      f'max={last_persistence_stats["max"]:.3f} '
                      f'min={last_persistence_stats["min"]:.3f} '
                      f'fb={last_persistence_stats["fallback"]}] '
                      f'ffl[on={int(self.use_ffl)} L={last_ffl_val:.5f} '
                      f'coef={last_ffl_coef:.3f} '
                      f'eff={last_ffl_coef * last_ffl_val:.5f} '
                      f'vs_lambda_pa*L_pa={self.res_loss_weight * l_pa.item():.5f}] '
                      f'L_fpa[on={int(self.use_freq_pa)} L={last_freqpa_val:.5f} '
                      f'coef={last_freqpa_coef:.3f} eff={last_freqpa_coef * last_freqpa_val:.5f}]')
                if self.opt.use_wandb:
                    self.wandb_logger('train/iter', **{
                        'mse_epct': mse_epct_val.item(),
                        'l_pa': l_pa.item(),
                        'epct_ramp': epct_ramp,
                        'gamma': gamma,
                        'res_loss_weight': self.res_loss_weight,
                        'chain_step': chain_step,
                        'l_pa_weight_std': last_persistence_stats['std'],
                        'l_pa_weight_max': last_persistence_stats['max'],
                        'l_pa_weight_min': last_persistence_stats['min'],
                        'l_pa_fallback': int(last_persistence_stats['fallback']),
                        'ffl_loss': last_ffl_val,
                        'ffl_coef': last_ffl_coef,
                        'ffl_effective': last_ffl_coef * last_ffl_val,
                        'l_fpa_loss': last_freqpa_val,
                        'l_fpa_coef': last_freqpa_coef,
                        'l_fpa_effective': last_freqpa_coef * last_freqpa_val,
                    })

        print('Logging epoch information...')
        epoch_log = {'loss': np.mean(losses), 'rmse': np.mean(rmses),
                     'ssim': np.mean(ssims), 'psnr': np.mean(psnrs)}
        current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        print(f'Epoch {self.epoch} lr={current_lr} loss={epoch_log["loss"]:.4f} '
              f'psnr={epoch_log["psnr"]:.4f}')
        if self.opt.use_wandb:
            self.wandb_logger('train/epoch', step_name='epoch', step=self.epoch, **epoch_log)
            self.wandb_logger('settings', step_name='epoch', step=self.epoch,
                              **{'current_lr': current_lr, 'batch_size': self.opt.batch_size})

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def val(self):
        self.net.eval()
        self.ema_net.eval()
        losses = []
        rmses, psnrs, ssims = [], [], []
        rmses_direct, psnrs_direct, ssims_direct = [], [], []
        ema_rmses, ema_psnrs, ema_ssims = [], [], []
        ema_rmses_direct, ema_psnrs_direct, ema_ssims_direct = [], [], []

        pbar = tqdm.tqdm(self.val_loader, ncols=60) if self.opt.use_tqdm else self.val_loader
        with torch.no_grad():
            for i, data in enumerate(pbar):
                mu_ct = data.to('cuda')
                num_views = self.net.view_list[-1]
                timestep_st = self.net.num_timesteps - 1
                sparse_mu, gt_mu = self.generate_sparse_and_gt_data(mu_ct, num_views=num_views)

                recon_mu, direct_recon_mu = self.net.sample(sparse_mu, timestep_st)
                recon_mu_ema, direct_recon_mu_ema = self.ema_net.sample(sparse_mu, timestep_st)

                loss = self.criterion(recon_mu, gt_mu)
                losses.append(loss.item())

                rmse, psnr, ssim = self.get_metrics_by_window(recon_mu, gt_mu)
                rmses.append(rmse); psnrs.append(psnr); ssims.append(ssim)

                rmse_d, psnr_d, ssim_d = self.get_metrics_by_window(direct_recon_mu, gt_mu)
                rmses_direct.append(rmse_d); psnrs_direct.append(psnr_d); ssims_direct.append(ssim_d)

                ema_rmse, ema_psnr, ema_ssim = self.get_metrics_by_window(recon_mu_ema, gt_mu)
                ema_rmses.append(ema_rmse); ema_psnrs.append(ema_psnr); ema_ssims.append(ema_ssim)

                ema_rmse_d, ema_psnr_d, ema_ssim_d = self.get_metrics_by_window(direct_recon_mu_ema, gt_mu)
                ema_rmses_direct.append(ema_rmse_d); ema_psnrs_direct.append(ema_psnr_d); ema_ssims_direct.append(ema_ssim_d)

        save_condition = np.mean(losses) < self.best_val_loss
        if save_condition:
            self.best_val_loss = np.mean(losses)

        print('Logging validation information...')
        epoch_log = {
            'loss': np.mean(losses),
            'rmse': np.mean(rmses), 'rmse_direct': np.mean(rmses_direct),
            'ssim': np.mean(ssims), 'ssim_direct': np.mean(ssims_direct),
            'psnr': np.mean(psnrs), 'psnr_direct': np.mean(psnrs_direct),
            'ema_rmse': np.mean(ema_rmses), 'ema_rmse_direct': np.mean(ema_rmses_direct),
            'ema_ssim': np.mean(ema_ssims), 'ema_ssim_direct': np.mean(ema_ssims_direct),
            'ema_psnr': np.mean(ema_psnrs), 'ema_psnr_direct': np.mean(ema_psnrs_direct),
        }
        print(f'Epoch {self.epoch} val psnr: {epoch_log["psnr"]:.4f} | ema_psnr: {epoch_log["ema_psnr"]:.4f}')
        if self.opt.use_wandb:
            self.wandb_logger('val/epoch', step_name='epoch', step=self.epoch, **epoch_log)

        return save_condition

    # ------------------------------------------------------------------
    # Fit loop
    # ------------------------------------------------------------------
    def fit(self):
        opt = self.opt
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)

        print(f'''Summary:
            Number of Epochs:      {opt.epochs}
            Batch Size:            {opt.batch_size}
            Initial Learning rate: {opt.lr}
            err_cfg_sigma:         {self.sigma}
            Training Size:         {len(self.train_dataset)}
            Validation Size:       {len(self.val_dataset)}
            Checkpoints Saved:     {opt.checkpoint_dir}
        ''')

        if self.opt.resume:
            assert self.opt.net_checkpath, "net_checkpath required for resume"
            self.net = self.load_model(net=self.net, net_checkpath=self.opt.net_checkpath, output=True)
        else:
            try:
                self.weights_init(self.net)
            except Exception as err:
                print(f'init failed: {err}')

        self.net = self.net.to(device)
        self.ema_net = self.ema_net.to(device)

        self.reset_parameters()
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=opt.batch_size,
            num_workers=opt.num_workers, pin_memory=True, shuffle=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=1, num_workers=opt.num_workers,
        )

        self.optimizer = self.get_optimizer(self.net)
        self.scheduler = self.get_scheduler(self.optimizer)

        self.iter = 0   # default; overridden by resume_opt if --resume_opt is set
        if self.opt.resume_opt:
            self.resume_opt()   # restores self.epoch and self.iter from saved opt ckpt
            print(f'resumed optimizers at epoch {self.epoch}.')

        if self.opt.resume:
            ema_ckpt = getattr(self.opt, 'ema_checkpath', '')
            if ema_ckpt:
                self.load_model(net=self.ema_net, net_checkpath=ema_ckpt)
                print('EMA net loaded from checkpoint')

        start_epoch = self.epoch
        for self.epoch in range(start_epoch, opt.epochs):
            print(f'start training epoch: {self.epoch}')
            self.train()
            if self.scheduler is not None:
                self.scheduler.step()
            save_condition = ((self.epoch + 1) % self.opt.save_epochs == 0) or \
                             ((self.epoch + 1) == self.opt.epochs)
            if save_condition:
                self.save_model(net=self.net, net_name='rbgdiff', ddp_model=False)
                self.save_model(net=self.ema_net, net_name='rbgdiff_ema')
                if self.epoch >= 5:
                    self.save_opt(optimizer=self.optimizer, scheduler=self.scheduler, opt_name='rbgdiff')
            if self.epoch % self.opt.val_interval == 0 or (self.epoch + 1) == self.opt.epochs:
                val_save_condition = self.val()
                if val_save_condition:
                    self.save_model(net=self.net, net_name='rbgdiff', best_val_model=True, ddp_model=False)
                    self.save_model(net=self.ema_net, net_name='rbgdiff_ema')
