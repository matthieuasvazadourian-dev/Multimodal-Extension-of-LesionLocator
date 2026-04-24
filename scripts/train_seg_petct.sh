set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate lesionlocator
export PATH="/home/runai-home/.local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
python -m pip install -e . --quiet

FOLD=${1:?"Usage: $0 <fold>"}

# Paths — PET+CT dataset lives in home (symlinks to scratch NIfTIs, zero extra disk usage)
TRAIN_DATA=/home/masva/datasets/Dataset900_USZMelanomaPETCT/imagesTr
TRAIN_PROMPT=/home/masva/datasets/Dataset900_USZMelanomaPETCT/labelsTr
VAL_DATA=/home/masva/datasets/Dataset901_USZMelanomaPETCT/imagesTr
VAL_PROMPT=/home/masva/datasets/Dataset901_USZMelanomaPETCT/labelsTr
CKPT_IN=/scratch/LesionLocator_saved_ckpt/TrainSeg800_LesionLocatorFTDec
CKPT_OUT=/home/masva/ckpt/TrainSeg900_PetCT_EarlyFusion
OUTPUT=/home/masva/ckpt/TrainSeg900_PetCT_EarlyFusion/fold_$FOLD

mkdir -p "$OUTPUT"
mkdir -p "$CKPT_OUT"

# Memory tuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

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
  --epochs 50 \
  --batch_size 1 \
  --lr 5e-5 \
  --num_workers 2 \
  --finetune first_conv \
  --train_fold $FOLD \
  --ckpt_path $CKPT_OUT \
  -npp 2 \
  -nps 2 \
  -device cuda \
  2>&1 | tee "$OUTPUT/train_seg_pet_fold_$FOLD.txt"
