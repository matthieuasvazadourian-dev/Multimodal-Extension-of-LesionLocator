#!/bin/bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate lesionlocator
export PATH="/home/runai-home/.local/bin:$PATH"

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
python -m pip install -e . --quiet

# Pass the checkpoint path as $1, e.g.:
#   ./scripts/display_fusion_weights.sh \
#     /home/masva/ckpt/TrainSeg900_Intermediate_Weighted/fold_0/checkpoint_final.pth
CKPT="${1:?usage: display_fusion_weights.sh <checkpoint.pth>}"

python -m lesionlocator.utilities.display_fusion_weights --ckpt "$CKPT"
