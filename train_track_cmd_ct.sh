#!/bin/bash
# filepath: /home/xiachen/scripts/train_track_cmd.sh
conda init
source ~/.bashrc
conda activate lesionlocator

cd /home/xiachen/scripts/LETITIA-LesionTracking
pip install -e .

cd /home/xiachen/scripts

FOLD=$1 # Accept fold number as a command-line argument

# Set PyTorch CUDA memory allocation config for better fragmentation handling
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# Run LesionLocator tracking training with specified arguments
# LesionLocator_train_track \
#     -bl /scratch/nnUNet_raw/Dataset800_USZMelanoma/imagesTr/TP0* \
#     -fu /scratch/nnUNet_raw/Dataset800_USZMelanoma/imagesTr/TP1* \
#     -pbl /scratch/nnUNet_raw/Dataset800_USZMelanoma/labelsTr/TP0* \
#     -pfu /scratch/nnUNet_raw/Dataset800_USZMelanoma/labelsTr/TP1* \
#     -tbl /scratch/nnUNet_raw/Dataset801_USZMelanoma/imagesTr/TP0* \
#     -tfu /scratch/nnUNet_raw/Dataset801_USZMelanoma/imagesTr/TP1* \
#     -tpbl /scratch/nnUNet_raw/Dataset801_USZMelanoma/labelsTr/TP0* \
#     -tpfu /scratch/nnUNet_raw/Dataset801_USZMelanoma/labelsTr/TP1* \
#     -o /home/xiachen/scripts/ckpt/TrainTrack800_FTDec/fold_$FOLD \
#     -t point \
#     -m /scratch/LesionLocator_saved_ckpt/TrainSeg800_LesionLocatorFTDec \
#     -f $FOLD \
#     --modality ct \
#     --epochs 10 \
#     --batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --num_workers 1 \
#     --finetune reg_net \
#     --train_fold $FOLD \
#     --ckpt_path /home/xiachen/scripts/ckpt/inference_checkpoints_ct \
#     -npp 1 \
#     -nps 1 \
#     -device cuda \
#     --visualize 2>&1 | tee /home/xiachen/scripts/train_track_ct_regnet_fold_$FOLD.txt

###
# /home/xiachen/scripts/ckpt/TrainTrack800_FTDec/fold_$FOLD - output path
# /scratch/LesionLocator_saved_ckpt/TrainSeg800_LesionLocatorFTDec - path to load during inference
###

LesionLocator_train_track \
  -i /scratch/nnUNet_raw/Dataset800_USZMelanoma/imagesTr \
  -iv /scratch/nnUNet_raw/Dataset801_USZMelanoma/imagesTr \
  -p /scratch/nnUNet_raw/Dataset800_USZMelanoma/labelsTr \
  -pv /scratch/nnUNet_raw/Dataset801_USZMelanoma/labelsTr \
  -o /home/xiachen/scripts/ckpt/TrainTrack800_FTDec/fold_$FOLD \
  -t point \
  -m /scratch/LesionLocator_saved_ckpt/TrainSeg800_LesionLocatorFTDec \
  -f $FOLD \
  --modality ct \
  --epochs 1 \
  --batch_size 1 \
  --lr 5e-5 \
  --gradient_accumulation_steps 1 \
  --num_workers 1 \
  --finetune unet \
  --train_fold $FOLD \
  --ckpt_path /home/xiachen/scripts/ckpt/inference_checkpoints_ct \
  -npp 2 \
  -nps 2 \
  -device cuda \
  --visualize 2>&1 | tee /home/xiachen/scripts/train_track_ct_regnet_fold_$FOLD.txt
