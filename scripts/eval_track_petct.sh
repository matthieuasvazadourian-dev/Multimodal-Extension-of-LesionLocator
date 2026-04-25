set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate lesionlocator
export PATH="/home/runai-home/.local/bin:$PATH"

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
python -m pip install -e . --quiet

FOLD=${1:?"Usage: $0 <fold> [empty_prompt]"}
EMPTY_PROMPT=${2:-"False"}

# Paths 
TEST_DATA=/scratch/nnUNet_raw/Dataset901_USZMelanoma/imagesTr
TEST_PROMPT=/scratch/nnUNet_raw/Dataset901_USZMelanoma/labelsTr
SEG_CKPT_ROOT=/home/masva/ckpt/TrainSeg900_PetCT_EarlyFusion
TRACK_CKPT_ROOT=/scratch/LesionLocator_saved_ckpt/TrainTrack800_FTDec
OUTPUT=/home/masva/vis_pet_track_eval/fold_$FOLD
COMBINED_CKPT_ROOT="$OUTPUT/combined_ckpt_root"

mkdir -p "$OUTPUT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

if [ -d "$SEG_CKPT_ROOT/LesionLocatorSeg" ]; then
    SEG_COMPONENT="$SEG_CKPT_ROOT/LesionLocatorSeg"
elif [ "$(basename "$SEG_CKPT_ROOT")" = "LesionLocatorSeg" ]; then
    SEG_COMPONENT="$SEG_CKPT_ROOT"
else
    echo "Could not locate LesionLocatorSeg inside: $SEG_CKPT_ROOT" >&2
    exit 1
fi

if [ -d "$TRACK_CKPT_ROOT/LesionLocatorTrack" ]; then
    TRACK_COMPONENT="$TRACK_CKPT_ROOT/LesionLocatorTrack"
elif [ "$(basename "$TRACK_CKPT_ROOT")" = "LesionLocatorTrack" ]; then
    TRACK_COMPONENT="$TRACK_CKPT_ROOT"
else
    echo "Could not locate LesionLocatorTrack inside: $TRACK_CKPT_ROOT" >&2
    exit 1
fi

mkdir -p "$COMBINED_CKPT_ROOT"
ln -sfn "$SEG_COMPONENT" "$COMBINED_CKPT_ROOT/LesionLocatorSeg"
ln -sfn "$TRACK_COMPONENT" "$COMBINED_CKPT_ROOT/LesionLocatorTrack"

if [ "$EMPTY_PROMPT" = "True" ]; then
    ADAPTIVE_FLAGS="--adaptive_mode --empty_prompt"
else
    ADAPTIVE_FLAGS="--adaptive_mode"
fi

LesionLocator_track_embed \
  -i  $TEST_DATA \
  -p  $TEST_PROMPT \
  -m  $COMBINED_CKPT_ROOT \
  -o  $OUTPUT \
  -f  $FOLD \
  -t  point \
  -npp 1 -nps 1 \
  --modality petct \
  --track \
  --lesion_focus \
  --crop_size 64 \
  $ADAPTIVE_FLAGS \
  2>&1 | tee "$OUTPUT/eval_track_pet_fold_$FOLD.txt"
