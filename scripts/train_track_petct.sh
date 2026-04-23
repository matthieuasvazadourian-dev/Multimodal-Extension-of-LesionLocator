set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate lesionlocator
export PATH="/home/runai-home/.local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
python -m pip install -e . --quiet

FOLD=${1:-0}
SEG_CKPT=/home/masva/ckpt/TrainSeg900_PetCT_EarlyFusion
OUTPUT_DIR=/home/masva/ckpt/train_track_pet/fold_${FOLD}
INFERENCE_CKPT_DIR=/home/masva/ckpt/inference

mkdir -p "$OUTPUT_DIR"
mkdir -p "$INFERENCE_CKPT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

echo "Fine-tuning TrackNet on Dataset900 (PET+CT), fold ${FOLD}..."

LesionLocator_train_track \
  -i  /scratch/nnUNet_raw/Dataset900_USZMelanoma/imagesTr \
  -iv /scratch/nnUNet_raw/Dataset901_USZMelanoma/imagesTr \
  -p  /scratch/nnUNet_raw/Dataset900_USZMelanoma/labelsTr \
  -pv /scratch/nnUNet_raw/Dataset901_USZMelanoma/labelsTr \
  -o  "$OUTPUT_DIR" \
  -t  point \
  -m  "$SEG_CKPT" \
  -f  "$FOLD" \
  --modality petct \
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
  2>&1 | tee "$OUTPUT_DIR/train_track_pet.txt"

echo "Done."
echo "  Raw checkpoints : $OUTPUT_DIR"
echo "  Inference ckpt  : $INFERENCE_CKPT_DIR"
