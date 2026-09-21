#!/usr/bin/env python3
"""Reproducible real-data preprocessing (Mayo Siemens): rebinned flat-fan projection
tif -> per-slice full-view mu recon (GT) -> save as HU int16 .npy (AAPM-format,
directly reusable by datasets.aapm.AAPMMyoDataset and train_real.py / eval_real.py).

    Train cases -> <out>/train_img/<case>_z<zzz>.npy  (one slice per file, like AAPM train_img)
    Test cases  -> <out>/test_vol/<case>.npy          (nz,256,256 int16 HU full-view GT), for per-case eval

Full-view recon uses wrappers/geometry_real.full_recon_from_sino (detector flip +
scale + REAL acquisition angles + real-geom FBP).

DATA / DEPENDENCIES ARE NOT BUNDLED WITH THE REPO. To run this you need:
  * The rebinned flat-fan projection tifs (<case>_flat_fan_projections.tif) produced from
    the Mayo Low-Dose CT Siemens raw projection cases (Helix2Fan flat-fan rebinning).
  * Helix2Fan (helper.load_tiff_stack_with_metadata) importable on sys.path.
  * torch_radon + a CUDA GPU.
Point --tif_dir at the rebinned tifs, --out at the output root, and --helix2fan at
the Helix2Fan-Modern dir. --split_csv defaults to configs/realdata_split.csv.

Built-in checkpoints (per case, logged to <out>/preprocess_manifest.csv):
  CP0.4  forward-NaN check on a mid slice -> SKIP case if NaN (never auto-shift).
  CP1.1  HU sanity (air~-1000, soft in range) per case (mid slice) -> flag.
  CP1.2  no NaN/inf in stored recon.
  CP1.4  sparse@72 FBP vs full-view PSNR in a sane band (sparse degradation works).
Deterministic + resumable (skips slices already written). Exit 0 iff no case FAILED a hard check.
"""
import os
import sys
import csv
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def parse_args():
    REPO = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="RBG-Diff real-data preprocessing (Siemens)")
    p.add_argument("--gpu", type=str, default="7")
    p.add_argument("--tif_dir", type=str, required=True, help="dir of <case>_flat_fan_projections.tif")
    p.add_argument("--out", type=str, required=True, help="output root (creates train_img/ and test_vol/)")
    p.add_argument("--helix2fan", type=str, required=True, help="Helix2Fan-Modern dir (provides helper.py)")
    p.add_argument("--split_csv", type=str, default=os.path.join(REPO, "configs", "realdata_split.csv"))
    p.add_argument("--config_dir", type=str, default=os.path.join(REPO, "configs"))
    return p.parse_args()


ARGS = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu

import numpy as np
import torch

sys.path.insert(0, ARGS.helix2fan)
_ = torch.zeros(1, device="cuda")

from wrappers import geometry_real as G
from helper import load_tiff_stack_with_metadata

DEV = "cuda"
OUT = ARGS.out
os.makedirs(f"{OUT}/train_img", exist_ok=True)
os.makedirs(f"{OUT}/test_vol", exist_ok=True)
GEOM = {"abd": G.load_geom(os.path.join(ARGS.config_dir, "geom_siemens_abdomen.json")),
        "chest": G.load_geom(os.path.join(ARGS.config_dir, "geom_siemens_chest.json"))}


def psnr(a, b, peak):
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse < 1e-9 else 10 * np.log10(peak * peak / mse)


def win(hu, w=3000, c=500):
    return np.clip((hu - (c - w / 2)) / w, 0, 1)


def main():
    split = {}
    with open(ARGS.split_csv) as f:
        for r in csv.DictReader(f):
            split[r["case"]] = (r["split"], r["bodypart"])

    rows = []
    for case, (sp, bp) in sorted(split.items()):
        geom = GEOM[bp]
        tif = f"{ARGS.tif_dir}/{case}_flat_fan_projections.tif"
        if not os.path.exists(tif):
            rows.append((case, sp, bp, 0, 0, "", "", "", "NO_TIF"))
            print(f"[{case}] NO TIF -> skip")
            continue
        arr, meta = load_tiff_stack_with_metadata(Path(tif))
        rotview, nu, nz = arr.shape
        assert rotview == geom["rotview"], f"{case} rotview {rotview}!={geom['rotview']}"
        zc = nz // 2
        mu_c = G.full_recon_from_sino(geom, arr[:, :, zc], meta['angles'], device=DEV)
        if bool(torch.isnan(mu_c).any()) or G.forward_has_nan(geom, mu_c, geom["rotview"]):
            rows.append((case, sp, bp, nz, 0, "", "", "", "FWD_NAN_SKIP"))
            print(f"[{case}] forward NaN -> SKIP")
            continue
        huc = G.mu2HU(mu_c[0, 0].cpu().numpy())
        air = float(np.percentile(huc, 1))
        soft = float(np.median(huc[(huc > -200) & (huc < 200)]))
        cp11 = (-1150 < air < -850) and (-200 < soft < 200)
        ops = G.build_fanbeam(geom, 72)
        sp72 = ops.backprojection(ops.filter_sinogram(ops.forward(mu_c), "ram-lak"))
        hus = G.mu2HU(sp72[0, 0].cpu().numpy())
        p72 = psnr(win(hus), win(huc), 1.0)
        cp14 = 10 < p72 < 45
        status = "OK" if (cp11 and cp14) else ("FLAG_HU" if not cp11 else "FLAG_PSNR")

        nsaved = 0
        bad = 0
        if sp == "train":
            for z in range(nz):
                outp = f"{OUT}/train_img/{case}_z{z:04d}.npy"
                if os.path.exists(outp):
                    nsaved += 1
                    continue
                mu = G.full_recon_from_sino(geom, arr[:, :, z], meta['angles'], device=DEV)
                hu = G.mu2HU(mu[0, 0].cpu().numpy())
                if not np.isfinite(hu).all():
                    bad += 1
                    continue
                np.save(outp, np.clip(hu, -1024, 3071).astype(np.int16)[None])  # (1,256,256) like aapm
                nsaved += 1
        else:  # test -> per-case volume
            outp = f"{OUT}/test_vol/{case}.npy"
            if os.path.exists(outp):
                nsaved = np.load(outp).shape[0]
            else:
                vol = np.empty((nz, 256, 256), np.int16)
                for z in range(nz):
                    mu = G.full_recon_from_sino(geom, arr[:, :, z], meta['angles'], device=DEV)
                    hu = G.mu2HU(mu[0, 0].cpu().numpy())
                    if not np.isfinite(hu).all():
                        bad += 1
                        hu = np.nan_to_num(hu)
                    vol[z] = np.clip(hu, -1024, 3071).astype(np.int16)
                np.save(outp, vol)
                nsaved = nz
        rows.append((case, sp, bp, nz, nsaved, f"{air:.0f}", f"{soft:.0f}", f"{p72:.1f}",
                     status + (f"_BAD{bad}" if bad else "")))
        print(f"[{case}] {sp} {bp} nz={nz} saved={nsaved} air={air:.0f} soft={soft:.0f} "
              f"sparse72PSNR={p72:.1f} -> {status}")

    with open(f"{OUT}/preprocess_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "split", "bodypart", "nz", "n_saved", "hu_air", "hu_soft", "sparse72_psnr", "status"])
        w.writerows(rows)
    ntr = sum(r[4] for r in rows if r[1] == "train")
    hardfail = [r[0] for r in rows if r[8] in ("FWD_NAN_SKIP", "NO_TIF") or r[8].startswith("FLAG")]
    print(f"\nsaved {OUT}/preprocess_manifest.csv | train slices total = {ntr}")
    print(f"cases with flags: {hardfail if hardfail else 'none'}")
    sys.exit(0 if not hardfail else 1)


if __name__ == '__main__':
    main()
