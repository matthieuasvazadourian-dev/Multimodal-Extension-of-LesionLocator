set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate lesionlocator
export PATH="/home/runai-home/.local/bin:$PATH"

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
python -m pip install -e . --quiet

FOLD=${1:-0}
SAVE_EMBED=${2:-False}
EMPTY_PROMPT=${3:-False}

CHECKPOINT=/scratch/LesionLocator_saved_ckpt/TrainSeg800_LesionLocatorFTDec
OUTPUT_DIR=/scratch/outputs_matthieu/eval_track_ct/fold_${FOLD}
LAYER_NAME=decoder.stages.2
EMBED_OUTPUT_DIR=/scratch/outputs_matthieu/embeddings/ct/decoder_stage_2/fold_${FOLD}

mkdir -p "$OUTPUT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

if [ "$SAVE_EMBED" = "True" ]; then
    echo "Extracting embeddings to $EMBED_OUTPUT_DIR..."
    mkdir -p "$EMBED_OUTPUT_DIR"
    LesionLocator_track_embed \
        -i /scratch/nnUNet_raw/Dataset800_USZMelanoma/imagesTr \
        -p /scratch/nnUNet_raw/Dataset800_USZMelanoma/labelsTr \
        -m "$CHECKPOINT" \
        -o "$OUTPUT_DIR" \
        -f "$FOLD" \
        -t point \
        -npp 2 -nps 2 \
        --extract_embeddings \
        --adaptive_mode \
        --embedding_layers "$LAYER_NAME" \
        --embedding_output_folder "$EMBED_OUTPUT_DIR" \
        --modality ct \
        --lesion_focus \
        --crop_size 64 \
        --track \
        2>&1 | tee "$OUTPUT_DIR/eval_track_save_embeds.txt"
else
    if [ "$EMPTY_PROMPT" = "True" ]; then
        ADAPTIVE_MODE="--adaptive_mode --empty_prompt"
    else
        ADAPTIVE_MODE="--adaptive_mode"
    fi
    echo "Running CT tracking evaluation, fold ${FOLD}..."
    LesionLocator_track \
        -i /scratch/nnUNet_raw/Dataset801_USZMelanoma/imagesTr \
        -p /scratch/nnUNet_raw/Dataset801_USZMelanoma/labelsTr \
        -m "$CHECKPOINT" \
        -o "$OUTPUT_DIR" \
        -f "$FOLD" \
        -t point \
        -npp 1 -nps 1 \
        --visualize \
        $ADAPTIVE_MODE \
        --modality ct \
        --track \
        2>&1 | tee "$OUTPUT_DIR/eval_track.txt"
fi

echo "Done. Results in: $OUTPUT_DIR"
