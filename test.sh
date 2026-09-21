#!/bin/bash
# ============================================================================
# RBG-Diff : SIM evaluation with the shipped checkpoint.
#   Reproduces the SIM main table at 18 / 36 / 72 views.
#   Expected avg PSNR under window (3000,500): 40.78 / 44.59 / 47.98 dB.
#   Usage: bash test.sh <GPU>
# ============================================================================
set -e

# --- environment ---
CONDA_SH=$(find "$HOME" -maxdepth 4 -name conda.sh -path '*/profile.d/*' 2>/dev/null | head -1)
source "$CONDA_SH"; conda activate CvG-Diff
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# --- paths (EDIT THESE) ---
REPO=$(cd "$(dirname "$0")" && pwd)
DATA_ROOT=${DATA_ROOT:-/path/to/aapm16}          # <-- AAPM16 root (not bundled); expects test_img/
cd "$REPO"

GPU=${1:-0}
NET=RBG-Diff
CKPT="$REPO/checkpoints/rbgdiff_sim_ema19.pkl"
OUT="$REPO/results/rbgdiff_sim"
mkdir -p "$OUT"

if [ ! -f "$CKPT" ]; then echo "MISSING checkpoint: $CKPT"; exit 1; fi

for V in 18 36 72; do
    echo "[test $(date '+%F %T')] view=$V"
    CUDA_VISIBLE_DEVICES=$GPU python main.py \
        --dataset_path "$DATA_ROOT/test_img" \
        --dataset_shape 256 --unet_dim 128 --trainer_mode test --num_views $V \
        --net_checkpath "$CKPT" --network $NET \
        --tester_save_path "$OUT" \
        --tester_save_name "sim_${V}v" --err_cfg_sigma 1.0 \
        2>&1 | tee "$OUT/eval_${V}v.log"
done
echo "[test $(date '+%F %T')] DONE -> $OUT"
