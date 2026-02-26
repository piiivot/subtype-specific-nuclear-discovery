#!/bin/bash
# Training script for MIME
# (Multi-label Inference with Model-based Estimation)
#
# Usage:
#   1. Edit the variables in the "Configuration" section below.
#   2. Run: bash train.sh

set -e

# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────

# Path to the LMDB directory containing patch images
DATA_PATH="/path/to/lmdb"

# Path to the k-fold split CSV (columns: fold, case_id, class_name)
# The bundled fold_splits.csv is relative to this script's directory.
FOLD_CSV_PATH="$(dirname "$0")/fold_splits.csv"

# Number of folds and fold indices
NUM_FOLDS=10

# Lists of (except_fold, cal_fold) pairs to train.
# except_fold: held-out validation fold
# cal_fold:    calibration fold (excluded from training, used for conformal prediction)
# Add or remove pairs as needed.
EXCEPT_FOLD_IDX_LIST=(0 2 4 6 8)
CAL_FOLD_IDX_LIST=(1 3 5 7 9)

# Output base directory (a timestamped sub-directory is created automatically)
OUTPUT_BASE_DIR="output/mime"

# ─────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────

# MIME-specific
WARM_UP_EPOCHS=5
TAU=0.3
STEPS_PER_EPOCH=1000
PSEUDO_LABEL_UPDATE_BATCH_SIZE=4096
EVAL_BATCH_SIZE=4096

# Model
NUM_CLASSES=8
LATENT_DIM=512
IN_FEATURES=512
PRETRAINED=true

# Loss
KL_WEIGHT=1e-3

# Training
BATCH_SIZE=256
NUM_EPOCHS=60
SEED=0

# Optimizer
LEARNING_RATE=1e-5
WEIGHT_DECAY=1e-5

# Optional: initialise from a checkpoint
# INIT_CHECKPOINT="/path/to/checkpoint.pth"
# BACKBONE_INIT_CHECKPOINT="/path/to/backbone_checkpoint.pth"

# ─────────────────────────────────────────
# Build optional checkpoint argument
# ─────────────────────────────────────────

if [ -n "${INIT_CHECKPOINT}" ]; then
    CHECKPOINT_ARG="--init_checkpoint ${INIT_CHECKPOINT}"
elif [ -n "${BACKBONE_INIT_CHECKPOINT}" ]; then
    CHECKPOINT_ARG="--backbone_init_checkpoint ${BACKBONE_INIT_CHECKPOINT}"
else
    CHECKPOINT_ARG=""
fi

# ─────────────────────────────────────────
# Training loop over fold pairs
# ─────────────────────────────────────────

NUM_FOLD_PAIRS=${#EXCEPT_FOLD_IDX_LIST[@]}

for i in $(seq 0 $((NUM_FOLD_PAIRS - 1))); do
    EXCEPT_FOLD_IDX=${EXCEPT_FOLD_IDX_LIST[$i]}
    CAL_FOLD_IDX=${CAL_FOLD_IDX_LIST[$i]}

    OUTPUT_DIR="${OUTPUT_BASE_DIR}/tau_${TAU}/except_${EXCEPT_FOLD_IDX}_cal_${CAL_FOLD_IDX}"

    echo "========================================="
    echo "Training fold pair ${i}: EXCEPT=${EXCEPT_FOLD_IDX}, CAL=${CAL_FOLD_IDX}"
    echo "========================================="

    python train.py \
        --data_path              "${DATA_PATH}" \
        --fold_csv_path          "${FOLD_CSV_PATH}" \
        --except_fold_idx        ${EXCEPT_FOLD_IDX} \
        --cal_fold_idx           ${CAL_FOLD_IDX} \
        --num_folds              ${NUM_FOLDS} \
        --batch_size             ${BATCH_SIZE} \
        --num_epochs             ${NUM_EPOCHS} \
        --seed                   ${SEED} \
        --device                 "cuda:0" \
        --num_workers            4 \
        --save_interval          1 \
        --learning_rate          ${LEARNING_RATE} \
        --weight_decay           ${WEIGHT_DECAY} \
        --pretrained             ${PRETRAINED} \
        --num_classes            ${NUM_CLASSES} \
        --in_features            ${IN_FEATURES} \
        --latent_dim             ${LATENT_DIM} \
        --kl_weight              ${KL_WEIGHT} \
        --warm_up_epochs         ${WARM_UP_EPOCHS} \
        --tau                    ${TAU} \
        --steps_per_epoch        ${STEPS_PER_EPOCH} \
        --pseudo_label_update_batch_size ${PSEUDO_LABEL_UPDATE_BATCH_SIZE} \
        --eval_batch_size        ${EVAL_BATCH_SIZE} \
        --output_dir             "${OUTPUT_DIR}" \
        ${CHECKPOINT_ARG}

    echo "Fold pair ${i} (except=${EXCEPT_FOLD_IDX}, cal=${CAL_FOLD_IDX}) done."
    echo ""
done

echo "All fold pairs finished."
