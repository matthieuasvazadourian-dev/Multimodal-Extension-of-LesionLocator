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
OUTPUT_DIR=/scratch/outputs_matthieu/eval_seg_ct/fold_${FOLD}

mkdir -p "$OUTPUT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

echo "Running CT segmentation evaluation (no tracking), fold ${FOLD}..."

LesionLocator_track \
    -i /scratch/nnUNet_raw/Dataset801_USZMelanoma/imagesTr \
    -p /scratch/nnUNet_raw/Dataset801_USZMelanoma/labelsTr \
    -m "$CHECKPOINT" \
    -o "$OUTPUT_DIR" \
    -f "$FOLD" \
    -t point \
    -npp 1 -nps 1 \
    --visualize \
    --modality ct \
    2>&1 | tee "$OUTPUT_DIR/eval_seg.txt"

echo "Done. Results in: $OUTPUT_DIR"
