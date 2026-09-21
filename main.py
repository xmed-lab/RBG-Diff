"""RBG-Diff entry point (train / test).

Instantiates the RBGDiff model and dispatches to the persistence-aware trainer
or the tester. This is the single, paper-named entry point for RBG-Diff.
"""

import os
import random
import argparse
import numpy as np
import torch

from networks.rbgdiff import RBGDiff
from trainers.rbgdiff_trainer import RBGDiffTrainer
from trainers.tester import RBGDiffTester


def get_parser():
    parser = argparse.ArgumentParser(description='RBG-Diff — sparse-view CT reconstruction')
    parser.add_argument('--log_interval', type=int, default=400)
    parser.add_argument('--val_interval', type=int, default=3)
    parser.add_argument('--checkpoint_root', type=str, default='')
    parser.add_argument('--checkpoint_dir', type=str, default='test')
    parser.add_argument('--use_tqdm', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='RBG-Diff')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--wandb_root', type=str, default='')
    parser.add_argument('--wandb_dir', type=str, default='')
    parser.add_argument('--wandb_api_key', type=str, default='')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', action='store_true', default=False)
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--dataset_name', default='aapm', type=str)
    parser.add_argument('--dataset_shape', type=int, default=512)
    parser.add_argument('--num_train', default=5410, type=int)
    parser.add_argument('--num_val', default=526, type=int)
    parser.add_argument('--split', default='test', type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--drop_last', default=False, type=bool)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--lr', default=4e-5, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--save_epochs', default=10, type=int)
    parser.add_argument('--scheduler', default='step', type=str)
    parser.add_argument('--step_size', default=25, type=int)
    parser.add_argument('--milestones', nargs='+', type=int)
    parser.add_argument('--step_gamma', default=1.0, type=float,
                        help='Default 1.0 (no decay).')
    parser.add_argument('--poly_iters', default=10, type=int)
    parser.add_argument('--poly_power', default=2, type=float)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--resume_opt', default=False, action='store_true')
    parser.add_argument('--net_checkpath', default='', type=str)
    parser.add_argument('--opt_checkpath', default='', type=str)
    parser.add_argument('--ema_checkpath', default='', type=str)
    parser.add_argument('--trainer_mode', default='train', type=str)
    parser.add_argument('--loss', default='l2', type=str)
    parser.add_argument('--network', default='RBG-Diff', type=str)
    parser.add_argument('--tester_save_name', default='default_save', type=str)
    parser.add_argument('--tester_save_image', default=False, action='store_true')
    parser.add_argument('--tester_save_path', default='', type=str)
    parser.add_argument('--num_views', default=18, type=int)
    parser.add_argument('--num_full_views', default=720, type=int)
    parser.add_argument('--unet_dim', type=int, default=128)
    parser.add_argument('--update_ema_iter', default=10, type=int)
    parser.add_argument('--start_ema_iter', default=2000, type=int)
    parser.add_argument('--ema_decay', default=0.995, type=float)
    parser.add_argument('--budget_ratio', type=int, default=2)
    parser.add_argument('--refine_budget', type=int, default=2)
    parser.add_argument('--time_back_ssim_threshold', type=float, default=0.98)
    parser.add_argument('--err_cfg_sigma', type=float, default=1.0)
    parser.add_argument('--epct_ramp_iters', type=int, default=None,
                        help='Iterations for cosine EPCT ramp 0->1. Defaults to start_ema_iter.')
    parser.add_argument('--res_dir', type=str, default='')
    parser.add_argument('--res_loss_weight', type=float, default=0.1,
                        help='Weight on the EPCT residual-branch loss.')

    # persistence-aware loss (trainer only)
    parser.add_argument('--gamma_max', type=float, default=8.0,
                        help='Max gamma for persistence**gamma weighting. Cosine ramp 0 -> gamma_max.')
    parser.add_argument('--gamma_ramp_iters', type=int, default=48000,
                        help='Iterations for cosine gamma ramp (~30 epochs at 1623 iters/epoch).')
    parser.add_argument('--step_weight_schedule', type=str, default='linear_late',
                        choices=['linear_late'],
                        help='Per-step persistence weighting schedule.')
    parser.add_argument('--use_ffl', action='store_true', default=False,
                        help='Enable Focal Frequency Loss on the residual head.')
    parser.add_argument('--ffl_weight', type=float, default=1.0,
                        help='FFL weight (absolute). Ramps with the persistence cosine schedule.')
    parser.add_argument('--ffl_alpha', type=float, default=1.0,
                        help='FFL focal exponent alpha (paper default 1.0).')
    parser.add_argument('--use_freq_pa', action='store_true', default=False,
                        help='Enable chain-persistent spectral focal loss (freq-pa) on res_out.')
    parser.add_argument('--freqpa_gamma_max', type=float, default=8.0,
                        help='Max focal gamma for freq_persistence weighting (cosine ramp 0->max).')
    parser.add_argument('--freqpa_weight', type=float, default=1.0,
                        help='freq-pa loss weight (ramps with the persistence cosine schedule).')
    parser.add_argument('--disable_bsrf', action='store_true', default=False,
                        help='(Ablation flag kept for the trainer API; the shipped RBG-Diff net is BSRF-on.)')
    return parser


def sparse_main(opt):
    if opt.epct_ramp_iters is None:
        opt.epct_ramp_iters = opt.start_ema_iter

    net_name = opt.network
    print('Network name:', net_name)

    if opt.res_dir and not opt.checkpoint_root:
        opt.checkpoint_root = os.path.join(opt.res_dir, 'ckpt')
    if opt.res_dir and not opt.wandb_root:
        opt.wandb_root = os.path.join(opt.res_dir, 'wandb')
    if not opt.checkpoint_dir or opt.checkpoint_dir == 'test':
        opt.checkpoint_dir = net_name
    if not opt.wandb_dir:
        opt.wandb_dir = net_name
    if not opt.run_name:
        opt.run_name = net_name

    wrapper_kwargs = {'num_full_views': opt.num_full_views, 'img_size': opt.dataset_shape}
    net = RBGDiff(opt, **wrapper_kwargs)

    if opt.trainer_mode == 'train':
        trainer = RBGDiffTrainer(opt=opt, net=net, loss_type=opt.loss)
        trainer.fit()
    elif opt.trainer_mode in ('test', 'val'):
        tester = RBGDiffTester(opt=opt, net=net, test_window=None)
        tester.iterative_sample()
    else:
        raise ValueError(f'trainer_mode must be train/val/test, got {opt.trainer_mode}')

    print('done')


if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()

    seed = 3407
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    sparse_main(opt)
