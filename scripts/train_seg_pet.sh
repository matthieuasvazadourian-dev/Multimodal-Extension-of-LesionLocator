#!/bin/bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate lesionlocator
export PATH="/home/runai-home/.local/bin:$PATH"

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
python -m pip install -e . --quiet

FOLD=${1:?"Usage: $0 <fold>"}

# NOTE: PET-only training expects a single-channel dataset where PET volumes
# are stored as _0000.nii.gz (not _0001.nii.gz as in Dataset900).
# Prepare Dataset902 by symlinking/copying the PET files from Dataset900
# and renaming them: case_XXXXX_0001.nii.gz -> case_XXXXX_0000.nii.gz

# Paths
TRAIN_DATA=/scratch/nnUNet_raw/Dataset902_USZMelanomaPET/imagesTr
TRAIN_PROMPT=/scratch/nnUNet_raw/Dataset902_USZMelanomaPET/labelsTr
VAL_DATA=/scratch/nnUNet_raw/Dataset903_USZMelanomaPET/imagesTr
VAL_PROMPT=/scratch/nnUNet_raw/Dataset903_USZMelanomaPET/labelsTr
# Pretrained CT checkpoint loads directly — 1-channel architecture matches
CKPT_IN=/scratch/LesionLocator_saved_ckpt/TrainSeg800_LesionLocatorFTDec
CKPT_OUT=/home/masva/ckpt/TrainSeg902_PET
OUTPUT=/home/masva/ckpt/TrainSeg902_PET/fold_$FOLD

mkdir -p "$OUTPUT"
mkdir -p "$CKPT_OUT"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

LesionLocator_train_segment \
  -i  $TRAIN_DATA \
  -p  $TRAIN_PROMPT \
  -iv $VAL_DATA \
  -pv $VAL_PROMPT \
  -o  $OUTPUT \
  -t  point \
  -m  $CKPT_IN \
  -f  $FOLD \
  --modality pet \
  --epochs 50 \
  --batch_size 1 \
  --lr 5e-5 \
  --num_workers 2 \
  --finetune decoder \
  --train_fold $FOLD \
  --ckpt_path $CKPT_OUT \
  -npp 2 \
  -nps 2 \
  -device cuda \
  2>&1 | tee "$OUTPUT/train_seg_pet_fold_$FOLD.txt"
