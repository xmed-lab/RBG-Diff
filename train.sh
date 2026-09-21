#!/bin/bash
# ============================================================================
# RBG-Diff : SIM training (from-scratch) of the final model
#   network  = RBG-Diff  (flat paper-named modules)
#   recipe   = the authoritative SIM main-table recipe (seed 3407, 20 epochs).
# Paths are repo-relative. Set REPO / DATA_ROOT below before running.
#   Usage: bash train.sh <GPU>
# ============================================================================
set -e

# --- environment ---
CONDA_SH=$(find "$HOME" -maxdepth 4 -name conda.sh -path '*/profile.d/*' 2>/dev/null | head -1)
source "$CONDA_SH"; conda activate CvG-Diff
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# --- paths (EDIT THESE) ---
REPO=$(cd "$(dirname "$0")" && pwd)
DATA_ROOT=${DATA_ROOT:-/path/to/aapm16}          # <-- AAPM16 root (not bundled); expects train_img/ and test_img/
cd "$REPO"

GPU=${1:-0}
TAG=rbgdiff_sim
NET=RBG-Diff
echo "[train $(date '+%F %T')] TAG=$TAG GPU=$GPU REPO=$REPO"

# --- TRAIN (20 epochs, seed 3407 hardcoded in main; BSRF on + pa + fpa) ---
CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=offline PYTHONUNBUFFERED=1 python -u main.py \
    --dataset_path "$DATA_ROOT/train_img" \
    --checkpoint_root "$REPO/logs" --checkpoint_dir $TAG \
    --dataset_shape 256 --unet_dim 128 --batch_size 2 --num_workers 4 \
    --num_full_views 720 \
    --epochs 20 --save_epochs 5 --val_interval 3 \
    --start_ema_iter 2000 --epct_ramp_iters 8000 --update_ema_iter 10 --ema_decay 0.995 \
    --lr 1e-4 --beta1 0.9 --beta2 0.999 --scheduler step --step_size 25 --step_gamma 1.0 \
    --err_cfg_sigma 1.0 --log_interval 100 --loss l2 \
    --use_wandb --wandb_project RBG-Diff --run_name $NET --local_rank 0 --use_tqdm \
    --res_loss_weight 0.1 --gamma_max 8.0 --gamma_ramp_iters 48000 --step_weight_schedule linear_late \
    --use_freq_pa --freqpa_gamma_max 8.0 --freqpa_weight 0.1 \
    2>&1 | tee "$REPO/logs/train_${TAG}.log"

echo "[train $(date '+%F %T')] DONE -> $REPO/logs/$TAG"
echo "The EMA-19 checkpoint (${TAG}-net-rbgdiff_ema_19_epoch.pkl) is the released weight."
