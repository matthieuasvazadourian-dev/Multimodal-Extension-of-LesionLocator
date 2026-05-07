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
CHECKPOINT=/scratch/LesionLocator_saved_ckpt/TrainSeg800_LesionLocatorFTDec
# Dataset901_CTonly: _0000-only symlinks from Dataset901 imagesTr (same 62 cases as petct eval)
# CT images are symlinks to Dataset801; using Dataset901 labelsTr ensures identical ground truth
TEST_DATA=/home/masva/datasets/Dataset901_CTonly/imagesTr
TEST_PROMPT=/home/masva/datasets/Dataset901_USZMelanomaPETCT/labelsTr
OUTPUT_DIR=/home/masva/vis_ct_seg_eval/fold_${FOLD}

mkdir -p "$OUTPUT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export CUDA_VISIBLE_DEVICES=0

LesionLocator_track \
    -i  $TEST_DATA \
    -p  $TEST_PROMPT \
    -m  "$CHECKPOINT" \
    -o  "$OUTPUT_DIR" \
    -f  "$FOLD" \
    -t  point \
    -npp 6 -nps 3 \
    --modality ct \
    2>&1 | tee "$OUTPUT_DIR/eval_seg_ct_fold_$FOLD.txt"
