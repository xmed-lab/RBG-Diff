# RBG-Diff: Residual-Bootstrapping Generalized Diffusion for Sparse-View CT Reconstruction

[![🤗 Dataset](https://img.shields.io/badge/🤗%20Dataset-RBG--Diff--SVCT--data-yellow)](https://huggingface.co/datasets/HajihajihaJimmy/RBG-Diff-SVCT-data)
[![🤗 Model](https://img.shields.io/badge/🤗%20Model-RBG--Diff-yellow)](https://huggingface.co/HajihajihaJimmy/RBG-Diff)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **RBG-Diff** (Residual-Bootstrapping Generalized Diffusion)
for sparse-view CT (SVCT) reconstruction. Existing generalized diffusion models follow
the global signal-to-noise-ratio pattern rather than the residual error, leaving
persistently hard regions under-corrected across iterative refinement. RBG-Diff casts
generalized-diffusion training as a **bootstrapped residual learning** problem over an
iterative SNR-biased reconstruction: at each step it estimates the residual error left by
the previous step and uses it to steer the current reconstruction toward the neglected,
error-prone regions.

## :rocket: Updates

- **[2026-09]** Code, pretrained checkpoints (simulation + real), and the preprocessed
  dataset are released.

## :star: Highlights

- **Bootstrapped residual learning.** Prior generalized diffusion models follow the
  global signal-to-noise-ratio pattern and leave persistently hard regions
  under-corrected across iterative refinement. RBG-Diff instead casts generalized-diffusion
  training as a bootstrapped residual learning problem over an iterative SNR-biased
  reconstruction, so each step estimates and corrects the residual error left behind.
- **Two residual cues, dual branches.** A reconstruction branch **R_θ** predicts the
  clean CT while a residual branch **R_ψ** estimates the leftover error from an
  input-anchored residual and the propagated residual, steering reconstruction toward the
  neglected regions.
- **Persistence-aware guidance + BSRF.** A spatial loss **L_pa** and a spectral loss
  **L_fpa** keep learning focused on persistently error-prone pixels and frequency bands,
  and **Band-Selective Residual Fusion (BSRF)** reweights frequency bands per
  (decoder-level, channel) at multiple scales.
- **Superior on both simulation and real scanner data.** RBG-Diff attains the best
  PSNR / SSIM / VIF across 18 / 36 / 72 views on simulated AAPM, and outperforms
  state-of-the-art generalized-diffusion baselines on real clinical scanner data.

## :hammer: Environment

RBG-Diff runs on **Python 3.10** and **PyTorch 2.2.0 (CUDA 11.8)**.

```shell
conda create -n rbgdiff python=3.10 -y
conda activate rbgdiff
# install a PyTorch 2.2 build matching your CUDA first, then:
pip install -r requirements.txt
```

### torch_radon (required, offline build)

`torch_radon` provides the forward/back projection and sparse-view geometry used to
simulate sinograms on the fly. It is not on PyPI and is built from source with the
compatibility patch shipped in this repo:

```shell
git clone https://github.com/matteo-ronchetti/torch-radon.git
cd torch-radon
patch -p1 < /path/to/RBG-Diff/torch-radon_fix/torch-radon_fix.patch
python setup.py install
```

A CUDA toolkit matching your PyTorch build must be available at compile time.

## :computer: Prepare Dataset

A preprocessed copy of both datasets, in exactly the layout the code expects, is on the
Hugging Face Hub. The data is derived from the TCIA collection *Low Dose CT Image and
Projection Data* (see [Data source & license](#page_facing_up-data-source--license)).

| Dataset | Anatomy | Folder |
|---|---|---|
| AAPM (simulation) | abdomen | `aapm16/{train_img,test_img}` |
| Mayo Siemens (real) | abdomen + chest | `real_siemens/{train_img,test_vol}` |

```shell
HF_HUB_ENABLE_HF_TRANSFER=1 hf download HajihajihaJimmy/RBG-Diff-SVCT-data \
    --repo-type dataset --local-dir ./rbgdiff_data
```

Images are one HU `.npy` per slice (256×256); `test_vol` holds per-case full-view GT
volumes. Sparse-view sinograms are simulated on the fly from each reference image (the
reference/GT is the 720-view FBP reconstruction). Metrics use HU window (3000, 500).

To rebuild from the original data instead of the mirror, obtain it from TCIA under its
terms and run the preprocessing scripts (`preprocess_real.py` for the real path).

## :key: Training & Evaluation

### Simulation (AAPM)

Set `DATA_ROOT` in `train.sh` / `test.sh` (or export it) to the `aapm16/` folder, then:

```shell
bash train.sh <GPU>   # train
bash test.sh  <GPU>   # evaluate; writes results/rbgdiff_sim/sim_<V>v/
```

The training recipe is set in the script (seed 3407, lr 1e-4, 20 epochs, batch 2,
`unet_dim 128`, `num_full_views 720`, `step_gamma 1.0`, `ema_decay 0.995`, `L_pa`
weight 0.1, `L_fpa` on). The released weight is the EMA at epoch 19.

**Results** — RBG-Diff on the AAPM test set (526 held-out slices), HU window (3000, 500):

| Views | PSNR (dB) | SSIM (×100) | VIF (×100) |
|-------|-----------|-------------|------------|
| 18    | 40.78     | 96.65       | 71.64      |
| 36    | 44.59     | 98.23       | 81.06      |
| 72    | 47.98     | 99.09       | 88.49      |

### Real data (Mayo Siemens)

The same model transfers to real clinical projection data by swapping only the geometry:
`wrappers/geometry_real.py` builds `torch_radon` fan-beam operators with the real scanner
geometry (from `configs/geom_siemens_{abdomen,chest}.json`) and re-points a live net so
the whole chain runs in that geometry. Network, trainer, sampler, and losses are unchanged.

| Purpose | File |
|---|---|
| Geometry adaptation | `wrappers/geometry_real.py` |
| Geometry / split configs | `configs/geom_siemens_{abdomen,chest}.json`, `configs/geom_ge.json`, `configs/realdata_split.csv` |
| Preprocessing (tif → HU npy) | `preprocess_real.py` |
| Training | `train_real.py` / `train_real.sh` |
| Evaluation | `eval_real.py` |

```shell
# preprocess (raw projection data not bundled; rebin with Helix2Fan first)
python preprocess_real.py --gpu <GPU> --tif_dir <flat_fan_tifs> --out <OUT_ROOT> --helix2fan <Helix2Fan dir>

# train from scratch (same recipe as simulation)
bash train_real.sh <GPU> 20 ./rbgdiff_data/real_siemens/train_img

# evaluate (defaults to the shipped real checkpoint)
python eval_real.py --gpu <GPU> --test_vol_dir ./rbgdiff_data/real_siemens/test_vol
```

**Results** — RBG-Diff on the real Mayo Siemens test set (12 held-out patients,
5,310 slices per view; SOMATOM Definition AS+ / Flash), HU window (3000, 500):

| Views | PSNR (dB) | SSIM (×100) | VIF (×100) |
|-------|-----------|-------------|------------|
| 18    | 38.56     | 92.79       | 53.51      |
| 36    | 40.78     | 94.73       | 62.45      |
| 72    | 42.65     | 96.22       | 70.03      |

RBG-Diff is best on every metric and view count. Over the strongest generalized-diffusion
baseline (CvG-Diff) this is **+1.00 / +1.25 / +0.97 dB** PSNR and **+3.72 / +4.89 / +4.01**
VIF at 18 / 36 / 72 views; all 27 comparisons across the four generalized-diffusion methods
are significant (one-sided paired Wilcoxon, Holm–Bonferroni corrected, p < 0.001).

## :inbox_tray: Download Checkpoints

Pretrained checkpoints are hosted on Hugging Face and can be downloaded for direct
inference: [HajihajihaJimmy/RBG-Diff](https://huggingface.co/HajihajihaJimmy/RBG-Diff).

```shell
HF_HUB_ENABLE_HF_TRANSFER=1 hf download HajihajihaJimmy/RBG-Diff --local-dir ./checkpoints
```

| File | Setting | View counts |
|------|---------|-------------|
| `rbgdiff_sim_ema19.pkl` | simulation (AAPM) | 18 / 36 / 72 |
| `rbgdiff_real_ema19.pkl` | real (Mayo Siemens) | 18 / 36 / 72 |

They are not bundled in this Git repository (100MB/file limit); the commands above
place them under `checkpoints/`, where `test.sh` and `eval_real.py` pick them up by default.

## :page_facing_up: Data source & license

The **code** is released under the MIT License (see `LICENSE`).

The CT data comes from the TCIA collection **Low Dose CT Image and Projection Data
(LDCT-and-Projection-data)** (DOI [10.7937/9npb-2637](https://doi.org/10.7937/9npb-2637)),
which curates the AAPM Low Dose CT Grand Challenge data. The **chest** and
**abdomen/liver** components used here are licensed **CC BY 4.0** (the head component,
under NIH controlled access, is not used). The Hugging Face dataset above is a processed
(modified) redistribution under CC BY 4.0. If you use the data you must cite it and
acknowledge the funding:

> McCollough, C., Chen, B., Holmes III, D., Duan, X., Yu, Z., Yu, L., Leng, S.,
> Fletcher, J. (2020). *Low Dose CT Image and Projection Data (LDCT-and-Projection-data)*
> (Version 7) [dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/9npb-2637
>
> Data collection was supported by NIBIB grants EB017095 and EB017185.

## :blue_book: Citation

If you find this work useful, please cite:

```bibtex
@article{rbgdiff,
  title   = {RBG-Diff: Residual-Bootstrapping Generalized Diffusion for Sparse-View CT Reconstruction},
  author  = {},
  journal = {},
  year    = {2026}
}
```
<!-- Citation to be updated on publication. -->

## :beers: Acknowledgements

We thank the Mayo Clinic and TCIA for the LDCT-and-Projection-data collection and the
AAPM Low Dose CT Grand Challenge, the [torch-radon](https://github.com/matteo-ronchetti/torch-radon)
project for the projection operators, and [Helix2Fan](https://github.com/faebstn96/helix2fan)
for the helical-to-flat-fan rebinning used in the real-data pipeline.
