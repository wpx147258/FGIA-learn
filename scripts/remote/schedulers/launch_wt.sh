#!/usr/bin/env bash
cd /root/autodl-tmp/cub-bird || exit 1
export HF_ENDPOINT=https://hf-mirror.com
C="--config configs/baseline_config.yaml --set DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16 TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 TRAIN.cam_part_size=224 TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3 MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1 TRAIN.swa_start_frac=[0.5,0.6,0.7] TRAIN.cam_teacher=gt_head TRAIN.cam_crop_hard_label=true TRAIN.part_pad=0.8"
CUDA_VISIBLE_DEVICES=3 nohup /root/miniconda3/bin/python train.py $C TRAIN.cam_part_views=wing,tail TRAIN.cam_part_w=0.3 EXP_NAME=r50_gth_wt > logs/r50_gth_wt_train.log 2>&1 &
echo "wt pid $!"
CUDA_VISIBLE_DEVICES=1 nohup /root/miniconda3/bin/python train.py $C TRAIN.cam_part_views=wing TRAIN.cam_part_w=0.5 EXP_NAME=r50_gth_w_w05 > logs/r50_gth_w_w05_train.log 2>&1 &
echo "w05 pid $!"
