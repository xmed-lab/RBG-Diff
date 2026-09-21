"""Our real-data training entry for RBG-Diff (Mayo Siemens / GE).

Trains the multiscale RBG-Diff net with the persistence-aware EPCT trainer
(paloss + freq-pa, no noise-aug) from scratch on real full-view mu recons. After
the trainer is built (so the EMA net is already deep-copied), it re-points
net.radon / net.image_radon / num_full_views to the real fan geometry via
``geometry_real.patch_net_real`` on both ``trainer.net`` and ``trainer.ema_net``,
so the cold-diffusion chain degrades with the real geometry.

Siemens abdomen (G1) & chest (G2) share fan params (dso/dsd/du/nu); only rotview
differs (2304 vs 1152), which only affects the dense full-view round-trip
(negligible). One geometry (abdomen, rotview 2304) is used for all Siemens
training. Data = real full-view recons (HU int16 .npy, one slice per file) via
``--dataset_path`` (see ``preprocess_real.py``; data not bundled).

Reuses ``main.get_parser`` (so every simulated-training flag is available) and
adds ``--geom_json``.
"""
import os
import sys
import random

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

from main import get_parser
from networks.rbgdiff import RBGDiff
from trainers.rbgdiff_trainer import RBGDiffTrainer
from wrappers import geometry_real as G


def main():
    parser = get_parser()
    parser.add_argument('--geom_json', type=str, required=True,
                        help='real geometry json (single training geometry, e.g. configs/geom_siemens_abdomen.json)')
    opt = parser.parse_args()
    if opt.epct_ramp_iters is None:
        opt.epct_ramp_iters = opt.start_ema_iter

    # dirs (mirror main.sparse_main)
    net_name = opt.network
    if not opt.checkpoint_dir or opt.checkpoint_dir == 'test':
        opt.checkpoint_dir = net_name
    if not opt.wandb_dir:
        opt.wandb_dir = net_name
    if not opt.run_name:
        opt.run_name = net_name

    seed = 3407
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    geom = G.load_geom(opt.geom_json)
    print(f"[realdata] geometry = {geom['name']} rotview={geom['rotview']} nu={geom['nu']} "
          f"dso={geom['dso']} dsd={geom['dsd']} (fan params shared across Siemens bodyparts)")

    net = RBGDiff(opt, num_full_views=geom['rotview'], img_size=opt.dataset_shape)
    trainer = RBGDiffTrainer(opt=opt, net=net, loss_type=opt.loss)

    # ---- inject REAL geometry on BOTH net and ema_net (after deepcopy, before fit) ----
    G.patch_net_real(trainer.net, geom, num_full_views=geom['rotview'])
    G.patch_net_real(trainer.ema_net, geom, num_full_views=geom['rotview'])
    print(f"[realdata] patched net + ema_net to real geometry; num_full_views={trainer.net.num_full_views}")

    trainer.fit()
    print('done')


if __name__ == '__main__':
    main()
