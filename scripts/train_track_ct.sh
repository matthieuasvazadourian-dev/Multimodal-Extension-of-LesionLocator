#!/bin/bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate lesionlocator
export PATH="/home/runai-home/.local/bin:$PATH"

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
python -m pip install -e . --quiet

FOLD=${1:-0}
CHECKPOINT=/scratch/LesionLocator_saved_ckpt/TrainSeg800_LesionLocatorFTDec
OUTPUT_DIR=/home/masva/ckpt/train_track/fold_${FOLD}
INFERENCE_CKPT_DIR=/home/masva/ckpt/inference

mkdir -p "$OUTPUT_DIR"
mkdir -p "$INFERENCE_CKPT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "Fine-tuning TrackNet on Dataset800, fold ${FOLD}..."

LesionLocator_train_track \
  -i  /scratch/nnUNet_raw/Dataset800_USZMelanoma/imagesTr \
  -iv /scratch/nnUNet_raw/Dataset801_USZMelanoma/imagesTr \
  -p  /scratch/nnUNet_raw/Dataset800_USZMelanoma/labelsTr \
  -pv /scratch/nnUNet_raw/Dataset801_USZMelanoma/labelsTr \
  -o  "$OUTPUT_DIR" \
  -t  point \
  -m  "$CHECKPOINT" \
  -f  "$FOLD" \
  --modality ct \
  --epochs 1 \
  --batch_size 1 \
  --lr 5e-5 \
  --gradient_accumulation_steps 1 \
  --num_workers 1 \
  --finetune unet \
  --train_fold "$FOLD" \
  --ckpt_path "$INFERENCE_CKPT_DIR" \
  -npp 2 \
  -nps 2 \
  -device cuda \
  --visualize \
  2>&1 | tee "$OUTPUT_DIR/train_track.txt"

echo "Done."
echo "  Raw checkpoints : $OUTPUT_DIR"
echo "  Inference ckpt  : $INFERENCE_CKPT_DIR"
