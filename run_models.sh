#!/usr/bin/env bash

DATA=/home/jovyan/shares/SR006.nfs2/dudko/data_scripts/FOR_TEST
ROOT=/home/jovyan/shares/SR006.nfs2/dudko/tetris
REPO=$ROOT/tetris-sam2-fork

SCRIPT=$REPO/batch_scripts/evaluate_model_user.py
LOGDIR=$REPO/logs
mkdir -p $LOGDIR

COMMON="--print-ious --save-ious --datasets=FOR_TEST --n_workers=1 --iou-analysis --thresh=0.5 --user_inputs --images_dir=$DATA/images --prompts_dir=$DATA/user_masks --masks_dir=$DATA/masks"

CUDA_VISIBLE_DEVICES=2 python $SCRIPT NoBRS --family=sam --model_type=vit_b --checkpoint=$ROOT/MODEL_CHECKPOINTS/SAM/sam_vit_b_01ec64.pth $COMMON > $LOGDIR/sam_vit_b.log 2>&1 || true;
CUDA_VISIBLE_DEVICES=2 python $SCRIPT NoBRS --family=sam --model_type=vit_h --checkpoint=$ROOT/MODEL_CHECKPOINTS/SAM/sam_vit_h_4b8939.pth $COMMON > $LOGDIR/sam_vit_h.log 2>&1 || true;
CUDA_VISIBLE_DEVICES=2 python $SCRIPT NoBRS --family=sam --model_type=vit_l --checkpoint=$ROOT/MODEL_CHECKPOINTS/SAM/sam_vit_l_0b3195.pth $COMMON > $LOGDIR/sam_vit_l.log 2>&1 || true;

CUDA_VISIBLE_DEVICES=7 python $SCRIPT NoBRS --family=robustsam --model_type=vit_b --checkpoint=$ROOT/MODEL_CHECKPOINTS/RobustSAM/robustsam_checkpoint_b.pth $COMMON > $LOGDIR/robustsam_vit_b.log 2>&1 || true;
CUDA_VISIBLE_DEVICES=7 python $SCRIPT NoBRS --family=robustsam --model_type=vit_h --checkpoint=$ROOT/MODEL_CHECKPOINTS/RobustSAM/robustsam_checkpoint_h.pth $COMMON > $LOGDIR/robustsam_vit_h.log 2>&1 || true;
CUDA_VISIBLE_DEVICES=7 python $SCRIPT NoBRS --family=robustsam --model_type=vit_l --checkpoint=$ROOT/MODEL_CHECKPOINTS/RobustSAM/robustsam_checkpoint_l.pth $COMMON > $LOGDIR/robustsam_vit_l.log 2>&1 || true;

CUDA_VISIBLE_DEVICES=3 python $SCRIPT NoBRS --family=sam2 --model_type=t --checkpoint=$ROOT/MODEL_CHECKPOINTS/SAM2.1/sam2.1_hiera_tiny.pt $COMMON > $LOGDIR/sam2_t.log 2>&1 || true;
CUDA_VISIBLE_DEVICES=3 python $SCRIPT NoBRS --family=sam2 --model_type=l --checkpoint=$ROOT/MODEL_CHECKPOINTS/SAM2.1/sam2.1_hiera_large.pt $COMMON > $LOGDIR/sam2_l.log 2>&1 || true;
CUDA_VISIBLE_DEVICES=3 python $SCRIPT NoBRS --family=sam2 --model_type=s --checkpoint=$ROOT/MODEL_CHECKPOINTS/SAM2.1/sam2.1_hiera_small.pt $COMMON > $LOGDIR/sam2_s.log 2>&1 || true;
CUDA_VISIBLE_DEVICES=3 python $SCRIPT NoBRS --family=sam2 --model_type=b --checkpoint=$ROOT/MODEL_CHECKPOINTS/SAM2.1/sam2.1_hiera_base_plus.pt $COMMON > $LOGDIR/sam2_b+.log 2>&1 || true;

CUDA_VISIBLE_DEVICES=1 python $SCRIPT NoBRS --family=samhq --model_type=vit_b --checkpoint=$ROOT/MODEL_CHECKPOINTS/SAM-HQ/sam_hq_vit_b.pth $COMMON > $LOGDIR/samhq_b.log 2>&1 || true;
CUDA_VISIBLE_DEVICES=1 python $SCRIPT NoBRS --family=samhq --model_type=vit_h --checkpoint=$ROOT/MODEL_CHECKPOINTS/SAM-HQ/sam_hq_vit_h.pth $COMMON > $LOGDIR/samhq_h.log 2>&1 || true;
CUDA_VISIBLE_DEVICES=1 python $SCRIPT NoBRS --family=samhq --model_type=vit_l --checkpoint=$ROOT/MODEL_CHECKPOINTS/SAM-HQ/sam_hq_vit_l.pth $COMMON > $LOGDIR/samhq_l.log 2>&1 || true;

CUDA_VISIBLE_DEVICES=2 python $SCRIPT NoBRS --family=mobile_sam --model_type=vit_t --checkpoint=$ROOT/MODEL_CHECKPOINTS/MobileSAM/mobile_sam.pt $COMMON > $LOGDIR/mobile_sam_t.log 2>&1 || true;

CUDA_VISIBLE_DEVICES=4 python $SCRIPT NoBRS --family=medsam --model_type=vit_b --checkpoint=$ROOT/MODEL_CHECKPOINTS/MedSAM/medsam_vit_b.pth $COMMON > $LOGDIR/medsam.log 2>&1 || true;
CUDA_VISIBLE_DEVICES=4 python $SCRIPT NoBRS --family=sammed2d --model_type=vit_b --checkpoint=$ROOT/MODEL_CHECKPOINTS/SAMMed2d/sam-med2d_b.pth $COMMON > $LOGDIR/sammed2d.log 2>&1 || true;

