#!/usr/bin/env bash
cd /root/autodl-tmp/cub-bird || exit 1
export HF_ENDPOINT=https://hf-mirror.com
C="--config configs/baseline_config.yaml --set DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16 TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3 MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1 TRAIN.swa_start_frac=[0.5,0.6,0.7] MODEL.tokenlearner=true MODEL.tl_ent_target=3.0 MODEL.tl_js_w=0.5 MODEL.tl_js_target=0.35 TRAIN.tl_reg_w=0.5 TRAIN.tl_aux_w=0.5"
CUDA_VISIBLE_DEVICES=0 nohup /root/miniconda3/bin/python train.py $C MODEL.tl_replace=true MODEL.tl_dropout=0.0 EXP_NAME=r50_tl_ent3_aux > logs/r50_tl_ent3_aux_train.log 2>&1 &
echo "ent3_aux $!"
CUDA_VISIBLE_DEVICES=2 nohup /root/miniconda3/bin/python train.py $C EXP_NAME=r50_tl_ent3_aux_res > logs/r50_tl_ent3_aux_res_train.log 2>&1 &
echo "ent3_aux_res $!"
