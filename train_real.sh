#!/bin/bash
# Real-data (Mayo Siemens) from-scratch training for RBG-Diff — final-model recipe
# (multiscale net + persistence-aware EPCT: paloss + freq-pa, NO noise-aug), with the
# real fan geometry injected via wrappers/geometry_real.patch_net_real.
#
# Usage: bash train_real.sh [GPU] [EPOCHS] [DATA_DIR]
#   DATA_DIR defaults to the preprocessed real full-view slice dir (HU int16 .npy,
#   one slice per file; see preprocess_real.py). Data is NOT bundled with the repo.
#
# Recipe matches the paper: lr 1e-4, bs 2, 20 epochs, ema 0.995, start_ema 2000,
# epct_ramp 8000, step_gamma 1.0 (no lr decay), err_cfg_sigma 1.0, seed 3407,
# res_loss_weight 0.1, gamma_max 8.0, freq-pa on (freqpa_weight 0.1).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
HERE=$(cd "$(dirname "$0")" && pwd)

GPU=${1:-7}
EPOCHS=${2:-20}
DATA=${3:-/path/to/real_npy/train_img}   # preprocessed real full-view slices (not bundled); see preprocess_real.py

# auto split: 90/10 by slice count (contiguous, like AAPM)
N=$(ls "$DATA"/*.npy 2>/dev/null | wc -l)
NVAL=$(( N / 10 )); NTRAIN=$(( N - NVAL ))
echo "real train slices N=$N -> num_train=$NTRAIN num_val=$NVAL  GPU=$GPU epochs=$EPOCHS"

CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python -u "$HERE/train_real.py" \
    --geom_json "$HERE/configs/geom_siemens_abdomen.json" \
    --dataset_path "$DATA" \
    --checkpoint_root "$HERE/logs" \
    --checkpoint_dir rbgdiff_realdata_siemens \
    --network rbgdiff_realdata_siemens \
    --dataset_shape 256 \
    --num_train $NTRAIN --num_val $NVAL \
    --unet_dim 128 \
    --batch_size 2 \
    --num_workers 4 \
    --epochs $EPOCHS \
    --save_epochs 5 \
    --val_interval 3 \
    --start_ema_iter 2000 \
    --epct_ramp_iters 8000 \
    --update_ema_iter 10 \
    --ema_decay 0.995 \
    --lr 1e-4 \
    --beta1 0.9 --beta2 0.999 \
    --scheduler step --step_size 25 --step_gamma 1.0 \
    --err_cfg_sigma 1.0 \
    --log_interval 100 \
    --local_rank 0 \
    --use_tqdm \
    --res_loss_weight 0.1 \
    --gamma_max 8.0 \
    --gamma_ramp_iters 48000 \
    --step_weight_schedule linear_late \
    --use_freq_pa --freqpa_gamma_max 8.0 --freqpa_weight 0.1
