#!/bin/bash
# filepath: /home/xiachen/scripts/run_train_segment.sh
conda init
source ~/.bashrc
conda activate lesionlocator

export CUDA_VISIBLE_DEVICES=0

cd /home/xiachen/scripts/LETITIA-LesionTracking
pip install -e .
# pip uninstall -y torch torchvision
# #pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# # OUTPUT_DIR=/home/xiachen/scripts/vis_track_all_noada
# FOLD=$1 # Accept fold number as a command-line argument

OUTPUT_DIR=/home/xiachen/scripts/vis_track_ada_embeds_stage2_train
FOLD=$1 # Accept fold number as a command-line argument
SAVE_EMBED=$2 # Set to True to save embeddings, False to skip saving
EMPTY_PROMPT=$3 # Set to True to run with empty prompt, False to use original prompts
LAYER_NAME=decoder.stages.2 # Specify which layer's embeddings to save
EMBED_OUTPUT_DIR=/scratch/outputs_xiaoran/progression_prediction/embeddings_pet/decoder_stage_2_16/train # Directory to save extracted embeddings

TRAIN_LABEL="900"  # !! not supposed to be used for evaluation, just to extract embeddings
TEST_LABEL="901"

if [ ! -d $OUTPUT_DIR ]; then
    mkdir $OUTPUT_DIR
fi

# Set PyTorch CUDA memory allocation config for better fragmentation handling
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

cd $OUTPUT_DIR
echo "Trying with -npp 2 -nps 2 for better performance..."

if [ "$SAVE_EMBED" = "True" ]; then
    echo "Saving embeddings to $EMBED_OUTPUT_DIR..."
    LesionLocator_track_embed \
        -i /scratch/nnUNet_raw/Dataset${TRAIN_LABEL}_USZMelanoma/imagesTr \
        -p /scratch/nnUNet_raw/Dataset${TRAIN_LABEL}_USZMelanoma/labelsTr \
        -m /scratch/LesionLocator_saved_ckpt/TrainSeg900_LesionLocatorFTDec \
        -o $OUTPUT_DIR \
        -f $FOLD \
        -t point \
        -npp 2 -nps 2 --extract_embeddings --adaptive_mode --embedding_layers $LAYER_NAME \
        --embedding_output_folder $EMBED_OUTPUT_DIR \
        --modality pet \
        --lesion_focus \
        --crop_size 64 \
        --track 2>&1 | tee $OUTPUT_DIR/eval_track_save_embeds.txt
else
    if [ "$EMPTY_PROMPT" = "True" ]; then
        ADAPTIVE_MODE="--adaptive_mode --empty_prompt"
    else
        ADAPTIVE_MODE="--adaptive_mode"
    fi
    echo "Running standard tracking evaluation, results saved to $OUTPUT_DIR..."
    LesionLocator_track \
            -i /scratch/nnUNet_raw/Dataset${TEST_LABEL}_USZMelanoma/imagesTr \
            -p /scratch/nnUNet_raw/Dataset${TEST_LABEL}_USZMelanoma/labelsTr \
            -m /scratch/LesionLocator_saved_ckpt/TrainSeg900_LesionLocatorFTDec \
            -o $OUTPUT_DIR \
            -f $FOLD \
            -t point \
            -npp 1 -nps 1 --visualize $ADAPTIVE_MODE \
            --modality pet \
            --track \
            --lesion_focus \
            --crop_size 64  2>&1 | tee $OUTPUT_DIR/eval_track.txt

fi



# # Evaluate tracking with fine-tuned model
# LesionLocator_track \
#     -i /scratch/nnUNet_raw/Dataset901_USZMelanoma/imagesTr \
#     -p /scratch/nnUNet_raw/Dataset901_USZMelanoma/labelsTr \
#     -m /scratch/LesionLocator_saved_ckpt/TrainSeg900_LesionLocatorFTDec \
#     -f $FOLD \
#     -o $OUTPUT_DIR \
#     -t "point" -npp 1 -nps 1 --visualize \
#     --modality "pet" --track 2>&1 | tee $OUTPUT_DIR/eval_track.txt

# # Check if the command failed due to memory issues
# if [ $? -ne 0 ]; then
#     echo "Memory issues detected with -npp 2 -nps 2. Retrying with -npp 1 -nps 1..."
#     # Clear any potential GPU memory
#     python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
    
#     # Wait a moment for cleanup
#     sleep 2

#     LesionLocator_track \
#         -i /scratch/nnUNet_raw/Dataset901_USZMelanoma/imagesTr \
#         -p /scratch/nnUNet_raw/Dataset901_USZMelanoma/labelsTr \
#         -m /scratch/LesionLocator_saved_ckpt/TrainSeg900_LesionLocatorFTDec \
#         -f $FOLD \
#         -o $OUTPUT_DIR \
#         -t "point" -npp 1 -nps 1 --visualize \
#         --modality "pet" --track 2>&1 | tee $OUTPUT_DIR/eval_track.txt
    
#     if [ $? -eq 0 ]; then
#         echo "Successfully completed with fallback settings -npp 1 -nps 1"
#     else
#         echo "Failed even with minimal process counts. Check system resources and error logs."
#         exit 1
#     fi
# else
#     echo "Successfully completed with -npp 2 -nps 2"
# fi