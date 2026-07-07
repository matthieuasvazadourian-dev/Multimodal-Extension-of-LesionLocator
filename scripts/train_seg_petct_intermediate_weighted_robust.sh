#!/bin/bash
set -euo pipefail

# Weighted intermediate fusion + full-fidelity ShaSpec (Wang et al., CVPR 2023)
# missing-modality robustness add-on. Same fusion_arch=weighted as
# train_seg_petct_intermediate_weighted.sh — --missing_modality_robust only
# adds two dedicated per-modality specific encoders, training-time modality
# dropout, and generator/combiner heads in front of it; it does not replace
# weighted fusion. This roughly doubles encoder-side compute/memory relative
# to weighted alone (deliberate quality-over-memory choice — see
# multimodal_unet.py docstring). If this OOMs, first try smaller batch/patch
# before reverting to a cheaper substitution-only design.
# To get the mcsa+robust combination instead, swap --fusion_arch to mcsa below.
# Usage: ./scripts/train_seg_petct_intermediate_weighted_robust.sh <fold>

source /opt/conda/etc/profile.d/conda.sh
conda activate lesionlocator
export PATH="/home/runai-home/.local/bin:$PATH"

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
python -m pip install -e . --quiet

FOLD=${1:?"Usage: $0 <fold>"}

TRAIN_DATA=/home/masva/datasets/Dataset900_USZMelanomaPETCT/imagesTr
TRAIN_PROMPT=/home/masva/datasets/Dataset900_USZMelanomaPETCT/labelsTr
VAL_DATA=/home/masva/datasets/Dataset901_USZMelanomaPETCT/imagesTr
VAL_PROMPT=/home/masva/datasets/Dataset901_USZMelanomaPETCT/labelsTr
CKPT_IN=/scratch/LesionLocator_saved_ckpt/TrainSeg800_LesionLocatorFTDec
CKPT_OUT=/home/masva/ckpt/TrainSeg900_Intermediate_WeightedRobust
OUTPUT=/home/masva/ckpt/TrainSeg900_Intermediate_WeightedRobust/fold_$FOLD

mkdir -p "$OUTPUT"
mkdir -p "$CKPT_OUT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
# NOTE: torch.compile is intentionally NOT enabled here — the network stashes
# the aux loss as a side-effect attribute each forward and branches on which
# modality was dropped, neither of which Dynamo traces reliably. The code also
# guards against compiling when missing_modality_robust is set.
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
  --fusion_arch weighted \
  --missing_modality_robust \
  --modality_dropout_p 0.5 \
  --lambda_da 0.1 \
  --lambda_dc 0.1 \
  --lambda_gen 0.1 \
  --epochs 50 \
  --batch_size 1 \
  --lr 5e-5 \
  --num_workers 2 \
  --finetune all \
  --train_fold $FOLD \
  --ckpt_path $CKPT_OUT \
  -npp 3 \
  -nps 2 \
  -device cuda \
  --cache \
  2>&1 | tee "$OUTPUT/train_seg_petct_weighted_robust_fold_$FOLD.txt"
