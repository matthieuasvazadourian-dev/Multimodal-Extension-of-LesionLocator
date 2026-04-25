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
OUTPUT_DIR=/home/masva/ckpt/train_seg/fold_${FOLD}
INFERENCE_CKPT_DIR=/home/masva/ckpt/inference

mkdir -p "$OUTPUT_DIR"
mkdir -p "$INFERENCE_CKPT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

echo "Fine-tuning segmentation model on Dataset800, fold ${FOLD}..."

LesionLocator_train_segment \
    -i  /scratch/nnUNet_raw/Dataset800_USZMelanoma/imagesTr \
    -iv /scratch/nnUNet_raw/Dataset801_USZMelanoma/imagesTr \
    -p  /scratch/nnUNet_raw/Dataset800_USZMelanoma/labelsTr \
    -pv /scratch/nnUNet_raw/Dataset801_USZMelanoma/labelsTr \
    -o  "$OUTPUT_DIR" \
    -t  point \
    -m  "$CHECKPOINT" \
    -f  "$FOLD" \
    --modality ct \
    --epochs 10 \
    --batch_size 1 \
    --lr 5e-5 \
    --num_workers 1 \
    --finetune decoder \
    --ckpt_path "$INFERENCE_CKPT_DIR" \
    --visualize \
    2>&1 | tee "$OUTPUT_DIR/train_seg.txt"

echo "Done."
echo "  Raw checkpoints : $OUTPUT_DIR"
echo "  Inference ckpt  : $INFERENCE_CKPT_DIR/LesionLocatorSeg/point_optimized/fold_${FOLD}/"
