set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate lesionlocator
export PATH="/home/runai-home/.local/bin:$PATH"

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
python -m pip install -e . --quiet

FOLD=${1:?"Usage: $0 <fold>"}

# Paths — PET+CT dataset lives in home (paired _0000 CT and _0001 PET files)
TEST_DATA=/home/masva/datasets/Dataset901_USZMelanomaPETCT/imagesTr
TEST_PROMPT=/home/masva/datasets/Dataset901_USZMelanomaPETCT/labelsTr
CKPT=/home/masva/ckpt/TrainSeg900_PetCT_EarlyFusion
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
  --modality petct \
  2>&1 | tee "$OUTPUT/eval_seg_pet_fold_$FOLD.txt"
