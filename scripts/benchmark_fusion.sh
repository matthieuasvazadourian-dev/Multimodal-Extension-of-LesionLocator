#!/bin/bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate lesionlocator
export PATH="/home/runai-home/.local/bin:$PATH"

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
python -m pip install -e . --quiet
pip install fvcore --quiet

PLANS=/scratch/LesionLocator_saved_ckpt/TrainSeg800_LesionLocatorFTDec/LesionLocatorSeg/point_optimized/plans.json
DATASET=/scratch/LesionLocator_saved_ckpt/TrainSeg800_LesionLocatorFTDec/LesionLocatorSeg/point_optimized/dataset.json
OUTPUT=/home/masva/benchmark_fusion.md

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m lesionlocator.utilities.benchmark_fusion \
  --plans   $PLANS \
  --dataset $DATASET \
  --patch   192 224 224 \
  --arches  weighted mcsa \
  --device  cuda \
  --n_warmup 5 \
  --n_iters  20 \
  --output  $OUTPUT \
  2>&1 | tee /home/masva/benchmark_fusion.txt
