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

# Paths — PET+CT dataset lives in home (symlinks to scratch NIfTIs, zero extra disk usage)
TRAIN_DATA=/home/masva/datasets/Dataset900_USZMelanomaPETCT/imagesTr
TRAIN_PROMPT=/home/masva/datasets/Dataset900_USZMelanomaPETCT/labelsTr
VAL_DATA=/home/masva/datasets/Dataset901_USZMelanomaPETCT/imagesTr
VAL_PROMPT=/home/masva/datasets/Dataset901_USZMelanomaPETCT/labelsTr
CKPT_IN=/scratch/LesionLocator_saved_ckpt/TrainSeg800_LesionLocatorFTDec
CKPT_OUT=/home/masva/ckpt/TrainSeg900_PetCT_EarlyFusion
OUTPUT=/home/masva/ckpt/TrainSeg900_PetCT_EarlyFusion/fold_$FOLD

mkdir -p "$OUTPUT"
mkdir -p "$CKPT_OUT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export LesionLocator_compile=1
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_MMAP_THRESHOLD_=65536

LesionLocator_train_segment \
  -i  $TRAIN_DATA \
  -p  $TRAIN_PROMPT \
  -iv $VAL_DATA \
  -pv $VAL_PROMPT \
  -o  $OUTPUT \
  -t  point \
  -m  $CKPT_IN \
  -f  $FOLD \
  --modality petct \
  --epochs 50 \
  --batch_size 1 \
  --lr 5e-5 \
  --num_workers 3 \
  --finetune first_conv \
  --train_fold $FOLD \
  --ckpt_path $CKPT_OUT \
  -npp 3 \
  -nps 2 \
  -device cuda \
  --cache \
  2>&1 | tee "$OUTPUT/train_seg_pet_fold_$FOLD.txt"
