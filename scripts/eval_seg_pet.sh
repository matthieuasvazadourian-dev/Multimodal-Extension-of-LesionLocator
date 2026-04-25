set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate lesionlocator
export PATH="/home/runai-home/.local/bin:$PATH"

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
python -m pip install -e . --quiet

FOLD=${1:?"Usage: $0 <fold>"}

# NOTE: expects PET-only dataset (Dataset903) where PET is stored as _0000.nii.gz

# Paths
TEST_DATA=/scratch/nnUNet_raw/Dataset903_USZMelanomaPET/imagesTr
TEST_PROMPT=/scratch/nnUNet_raw/Dataset903_USZMelanomaPET/labelsTr
CKPT=/home/masva/ckpt/TrainSeg902_PET
OUTPUT=/home/masva/vis_pet_seg_eval/fold_$FOLD

mkdir -p "$OUTPUT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
export CUDA_VISIBLE_DEVICES=0

LesionLocator_track \
  -i  $TEST_DATA \
  -p  $TEST_PROMPT \
  -m  $CKPT \
  -o  $OUTPUT \
  -f  $FOLD \
  -t  point \
  -npp 1 -nps 1 \
  --modality pet \
  2>&1 | tee "$OUTPUT/eval_seg_pet_fold_$FOLD.txt"
