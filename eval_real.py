#!/usr/bin/env python3
"""Our real-data evaluation for RBG-Diff on the Mayo Siemens test set. Uses the
shipped real checkpoint (or --ckpt), all held-out test cases (6 abdomen + 6
chest), NSL slices each, per view {72,36,18}:
    null PSNR/SSIM/VIF  (1-step direct recon)
    final PSNR/SSIM/VIF (iterative chain recon)
    B_psnr = final - null
Each case uses its own real scanner geometry. Window width=3000 / center=500.
Prints per-case + aggregate tables and writes a CSV.

The shipped real checkpoint loads directly; a checkpoint in the older key naming
is mapped automatically (tools/remap_ckpt).

Metrics (window 3000/500):
  * PSNR/SSIM_sk : skimage SSIM on [0,1], data_range=1
  * SSIM_cm      : utilities.metrics.compute_measure (win11 torch SSIM)
  * VIF          : torchmetrics visual_information_fidelity on [0,1]x255
"""
import os
import sys
import glob
import re
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def parse_args():
    RVAL = "/path/to/real_data"   # unbundled real data + training logs root; override paths below via CLI
    p = argparse.ArgumentParser(description="RBG-Diff real test-set eval")
    p.add_argument("--gpu", type=str, default="7")
    _here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--ckpt", type=str, default=os.path.join(_here, "checkpoints", "rbgdiff_real_ema19.pkl"),
                   help="ema checkpoint (.pkl); defaults to the shipped real ckpt. Pass --ckpt '' to pick the last ema in --ckpt_dir instead")
    p.add_argument("--ckpt_dir", type=str,
                   default=f"{RVAL}/logs/v3_13_lfa_fbm_multiscale_realdata_siemens",
                   help="dir to search for the last *ema_<N>_epoch.pkl if --ckpt is empty")
    p.add_argument("--test_vol_dir", type=str, default=f"{RVAL}/work/real_npy/test_vol",
                   help="dir of per-case full-view GT volumes <case>.npy (nz,256,256) int16 HU")
    p.add_argument("--config_dir", type=str, default="",
                   help="dir with geom_siemens_{abdomen,chest}.json (default: <repo>/configs)")
    p.add_argument("--views", type=int, nargs="+", default=[72, 36, 18])
    p.add_argument("--nsl", type=int, default=5, help="slices per case (evenly spaced in 0.35..0.65 depth)")
    p.add_argument("--cases", type=str, nargs="+", default=[],
                   help="subset of case ids to evaluate (default: all 12)")
    p.add_argument("--out_csv", type=str, default="", help="output csv path (default: <repo>/logs/...)")
    p.add_argument("--unet_dim", type=int, default=128)
    p.add_argument("--err_cfg_sigma", type=float, default=1.0)
    return p.parse_args()


ARGS = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu

import types
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn
from torchmetrics.functional.image import visual_information_fidelity as vif

_ = torch.zeros(1, device="cuda")

from networks.rbgdiff import RBGDiff
from wrappers import geometry_real as G
from tools.remap_ckpt import remap_state_dict
from utilities.metrics import compute_measure

DEV = "cuda"
REPO = os.path.dirname(os.path.abspath(__file__))
CFG = ARGS.config_dir if ARGS.config_dir else os.path.join(REPO, "configs")
GEOM = {"abd": G.load_geom(os.path.join(CFG, "geom_siemens_abdomen.json")),
        "chest": G.load_geom(os.path.join(CFG, "geom_siemens_chest.json"))}

# 12 held-out test cases (split test)
ALL_CASES = [("L237", "abd"), ("L057", "abd"), ("L186", "abd"), ("L248", "abd"), ("L145", "abd"), ("L273", "abd"),
             ("C158", "chest"), ("C296", "chest"), ("C160", "chest"), ("C267", "chest"), ("C111", "chest"), ("C012", "chest")]
if ARGS.cases:
    sel = set(ARGS.cases)
    CASES = [(c, b) for c, b in ALL_CASES if c in sel]
else:
    CASES = ALL_CASES
VIEWS = ARGS.views
NSL = ARGS.nsl


def win(h):
    return np.clip((h + 1000) / 3000.0, 0, 1)   # window width=3000 center=500 -> [-1000,2000]


def psnr(a, b):
    m = np.mean((win(a) - win(b)) ** 2)
    return 99.0 if m < 1e-9 else 10 * np.log10(1.0 / m)


def ssim(a, b):
    return float(ssim_fn(win(a), win(b), data_range=1.0))


def vif_m(a, b):
    p = torch.from_numpy(win(a) * 255.0)[None, None].float().to(DEV)
    g = torch.from_numpy(win(b) * 255.0)[None, None].float().to(DEV)
    return float(vif(p, g))


def ssim_cm(a, b):
    p = torch.from_numpy(win(a).astype(np.float32))[None, None].to(DEV)
    g = torch.from_numpy(win(b).astype(np.float32))[None, None].to(DEV)
    return float(compute_measure(p, g, 1)[2])


def load_real_ckpt(net, path):
    """Load an EMA checkpoint into RBGDiff, mapping legacy key names when present."""
    ck = torch.load(path, map_location="cpu")
    sd = ck.get("net_param", ck)
    is_new = any(k.startswith("denoise_fn.recon_") for k in sd)
    if not is_new:
        sd = remap_state_dict(sd)
    net.load_state_dict(sd, strict=True)
    return ck.get("epoch", None)


def epoch_of(p):
    m = re.search(r"ema_(\d+)_epoch", p)
    return int(m.group(1)) if m else -1


def main():
    if ARGS.ckpt:
        CK = ARGS.ckpt
        EP = epoch_of(CK)
    else:
        ckpts = sorted(glob.glob(f"{ARGS.ckpt_dir}/*ema_[0-9]*_epoch.pkl"), key=epoch_of)
        assert ckpts, f"no ema ckpt found in {ARGS.ckpt_dir}"
        CK = ckpts[-1]
        EP = epoch_of(CK)
    print(f"eval using ema checkpoint: epoch {EP}  ({os.path.basename(CK)})")

    opt = types.SimpleNamespace(unet_dim=ARGS.unet_dim, err_cfg_sigma=ARGS.err_cfg_sigma,
                                disable_bsrf=False)
    net = RBGDiff(opt, num_full_views=2304, img_size=256).to(DEV).eval()
    load_real_ckpt(net, CK)

    @torch.no_grad()
    def eval_slice(mu, geom, V):
        G.patch_net_real(net, geom, num_full_views=geom["rotview"])
        sparse, full = net.generate_sparse_and_full_ct(mu, num_views=V)
        final, nullr = net.sample_chain(sparse, net.view_list.index(V))
        gt = G.mu2HU(full[0, 0].cpu().numpy())
        hn = G.mu2HU(nullr[0, 0].cpu().numpy())
        hf = G.mu2HU(final[0, 0].cpu().numpy())
        return (psnr(hn, gt), psnr(hf, gt), ssim(hn, gt), ssim(hf, gt),
                vif_m(hn, gt), vif_m(hf, gt), ssim_cm(hn, gt), ssim_cm(hf, gt))

    rows = []
    percase = {}
    for case, bp in CASES:
        vol = np.load(f"{ARGS.test_vol_dir}/{case}.npy")
        zs = np.linspace(int(0.35 * vol.shape[0]), int(0.65 * vol.shape[0]), NSL).astype(int)
        mus = [torch.from_numpy(G.HU2mu(vol[z].astype(np.float32)))[None, None].to(DEV) for z in zs]
        for V in VIEWS:
            vals = np.array([eval_slice(mu, GEOM[bp], V) for mu in mus])
            m = vals.mean(0)
            percase[(case, V)] = m
            rows.append([case, bp, V, f"{m[0]:.2f}", f"{m[1]:.2f}", f"{m[1] - m[0]:+.2f}",
                         f"{m[2]:.4f}", f"{m[3]:.4f}", f"{m[4]:.4f}", f"{m[5]:.4f}",
                         f"{m[6]:.4f}", f"{m[7]:.4f}"])
        print(f"{case} done")

    out_csv = ARGS.out_csv if ARGS.out_csv else os.path.join(REPO, "logs", f"rbgdiff_realdata_test_ema{EP}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "bodypart", "view", "null_psnr", "final_psnr", "B_psnr",
                    "null_ssim_sk", "final_ssim_sk", "null_vif", "final_vif", "null_ssim_cm", "final_ssim_cm"])
        w.writerows(rows)

    print("\n" + "=" * 84)
    print(f"REAL TEST-SET (ema epoch {EP}, wide window 3000/500, {len(CASES)} cases x {NSL} slices)")
    print("=" * 84)
    print(f"{'view':>5} | {'null PSNR':>9} | {'final PSNR':>10} | {'B=f-n':>7} | "
          f"{'fin SSIM_sk':>11} | {'fin SSIM_cm':>11} | {'null VIF':>8} | {'final VIF':>9}")
    for V in VIEWS:
        arr = np.array([percase[(c, V)] for c, _ in CASES])
        m = arr.mean(0)
        print(f"{V:>5} | {m[0]:>9.2f} | {m[1]:>10.2f} | {m[1] - m[0]:>+7.2f} | "
              f"{m[3]:>11.4f} | {m[7]:>11.4f} | {m[4]:>8.4f} | {m[5]:>9.4f}")
    print(f"\nsaved {out_csv}")


if __name__ == '__main__':
    main()
