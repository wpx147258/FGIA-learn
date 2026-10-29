#!/usr/bin/env bash
cd /root/autodl-tmp/cub-bird || exit 1
export HF_ENDPOINT=https://hf-mirror.com
BASE="--config configs/baseline_config.yaml --set DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16 TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3 MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1 TRAIN.swa_start_frac=[0.5,0.6,0.7] MODEL.part_attn=true MODEL.pa_parts=8 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 MODEL.pa_depth=1 TRAIN.pa_div_w=1.0 TRAIN.pa_aux_w=0.25 TRAIN.pa_lr=5.0e-4"
CUDA_VISIBLE_DEVICES=1 nohup /root/miniconda3/bin/python train.py $BASE MODEL.pa_from=layer4 SEED=1 EXP_NAME=r50_pa4_k8_s1 > logs/r50_pa4_k8_s1_train.log 2>&1 &
echo "k8_s1 $!"
CUDA_VISIBLE_DEVICES=3 nohup /root/miniconda3/bin/python train.py $BASE MODEL.pa_from=layer3 EXP_NAME=r50_pa4_l3_k8 > logs/r50_pa4_l3_k8_train.log 2>&1 &
echo "l3_k8 $!"
