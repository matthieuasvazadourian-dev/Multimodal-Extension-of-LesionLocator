#!/bin/bash
set -euo pipefail

# Approach A: on/off switch for missing-modality robustness.
# Partitions a mixed PET/CT input dir per case by modality presence and dispatches
# each subset to the matching model:
#   CT + PET -> fusion model (petct)   CT only -> CT model   PET only -> PET model
# No core inference changes; pure orchestration over the existing entry points.
#
# Usage: ./scripts/eval_seg_missing_modality.sh <fold> [weighted|mcsa] [input_dir]

source /opt/conda/etc/profile.d/conda.sh
conda activate lesionlocator
export PATH="/home/runai-home/.local/bin:$PATH"

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
python -m pip install -e . --quiet

FOLD=${1:?"Usage: $0 <fold> [weighted|mcsa]"}
FUSION_ARCH=${2:-weighted}

# --- Config: input, prompts, checkpoints, output ------------------------------
INPUT=${3:-/home/masva/datasets/Dataset901_USZMelanomaPETCT/imagesTr}  # mixed dir to route
PROMPT=/home/masva/datasets/Dataset901_USZMelanomaPETCT/labelsTr  # labels keyed by case id

if [ "$FUSION_ARCH" = "mcsa" ]; then
  PETCT_CKPT=/home/masva/ckpt/TrainSeg900_Intermediate_MCSA
else
  PETCT_CKPT=/home/masva/ckpt/TrainSeg900_Intermediate_Weighted
fi
CT_CKPT=/scratch/LesionLocator_saved_ckpt/TrainSeg800_LesionLocatorFTDec
PET_CKPT=/home/masva/ckpt/TrainSeg902_PET

OUTPUT=/home/masva/vis_missing_modality_eval/fold_${FOLD}
ROUTE_TMP=${OUTPUT}/_route
# ------------------------------------------------------------------------------

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$OUTPUT"
rm -rf "$ROUTE_TMP"

echo "[route] Partitioning $INPUT by modality presence..."
python -m lesionlocator.utilities.route_by_modality --images "$INPUT" --outdir "$ROUTE_TMP"

run_subset () {   # $1=subset dir name  $2=modality  $3=checkpoint  $4=extra args
  local sub="$1" modality="$2" ckpt="$3" extra="$4"
  local imgs="$ROUTE_TMP/$sub/imagesTr"
  if [ ! -d "$imgs" ] || [ -z "$(ls -A "$imgs" 2>/dev/null)" ]; then
    echo "[$sub] no cases, skipping."
    return
  fi
  echo "[$sub] routing $(ls -A "$imgs" | wc -l) file(s) -> $modality model"
  LesionLocator_track \
    -i "$imgs" \
    -p "$PROMPT" \
    -m "$ckpt" \
    -o "$OUTPUT/$sub" \
    -f "$FOLD" \
    -t point \
    -npp 6 -nps 3 \
    --modality "$modality" $extra \
    2>&1 | tee "$OUTPUT/eval_${sub}_fold_${FOLD}.txt"
}

run_subset petct petct "$PETCT_CKPT" "--fusion_arch $FUSION_ARCH"
run_subset ct    ct    "$CT_CKPT"    ""
run_subset pet   pet   "$PET_CKPT"   ""

echo "[done] Per-subset predictions under $OUTPUT/{petct,ct,pet}"
