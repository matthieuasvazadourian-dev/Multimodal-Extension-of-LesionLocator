#!/bin/bash
set -euo pipefail

# Phase 6 "Job A" — measure missing-modality robustness of one weighted+robust
# checkpoint. Runs LesionLocator_track 3x on the SAME paired val set
# (Dataset901_USZMelanomaPETCT — both _0000/_0001 files present on disk for
# every case), using --force_inference_modality to make the network mask a
# channel internally rather than dropping it from the input (see
# multimodal_unet.py's inference_modality switch). This is the measurement
# path: it quantifies the robustness tradeoff on a dataset that never actually
# has a missing file. For genuine per-case missing files (Phase 7 deployment),
# use LesionLocator_track directly on a mixed folder without
# --force_inference_modality — the loader auto-detects each case's modality.
#
# To eval the mcsa+robust combination instead, swap --fusion_arch/CKPT below.
# Usage: ./scripts/eval_seg_missing_modality_robust.sh <fold>

source /opt/conda/etc/profile.d/conda.sh
conda activate lesionlocator
export PATH="/home/runai-home/.local/bin:$PATH"

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
python -m pip install -e . --quiet

FOLD=${1:?"Usage: $0 <fold>"}

TEST_DATA=/home/masva/datasets/Dataset901_USZMelanomaPETCT/imagesTr
TEST_PROMPT=/home/masva/datasets/Dataset901_USZMelanomaPETCT/labelsTr
CKPT=/home/masva/ckpt/TrainSeg900_Intermediate_WeightedRobust
OUTPUT_ROOT=/home/masva/vis_missing_modality_robust_eval/fold_$FOLD

mkdir -p "$OUTPUT_ROOT/both" "$OUTPUT_ROOT/ct_only" "$OUTPUT_ROOT/pet_only"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

run_scenario() {
  local subdir=$1
  shift
  LesionLocator_track \
    -i  $TEST_DATA \
    -p  $TEST_PROMPT \
    -m  $CKPT \
    -o  "$OUTPUT_ROOT/$subdir" \
    -f  $FOLD \
    -t  point \
    -npp 6 -nps 3 \
    --modality petct \
    --fusion_arch weighted \
    --missing_modality_robust \
    "$@" \
    2>&1 | tee "$OUTPUT_ROOT/$subdir/eval_log.txt"
}

echo "=== [1/3] Both modalities present (baseline) ==="
run_scenario both

echo "=== [2/3] Forced CT-only (PET masked internally, --force_inference_modality ct) ==="
run_scenario ct_only --force_inference_modality ct

echo "=== [3/3] Forced PET-only (CT masked internally, --force_inference_modality pet) ==="
run_scenario pet_only --force_inference_modality pet

echo "=== Dice/NSD summary ==="
python -m lesionlocator.utilities.diff_missing_modality_dice "$OUTPUT_ROOT" \
  | tee "$OUTPUT_ROOT/dice_summary_fold_$FOLD.txt"
