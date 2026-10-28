#!/usr/bin/env bash
# 优先跑「w=0.5 在 89.01 那个底座上」，之后接着跑 H（w=0.8 在 88.62 底座上）。
#
# 已有的权重证据：
#   88.62 底座（l4=0.1、p1.0）:  w0.15→88.46  w0.3→88.62  w0.5→88.80   w0.8→? (H)
#   89.01 底座（l4=0.05、p0.7/ls）: w0.3→89.01   w0.8→88.58   ← 0.3 和 0.8 之间有空档
# 所以先补 89.01 底座上的 w=0.5。
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/home/wpx/anaconda3/bin/python
LOG=outputs/_logs/queue_w05_first.log

echo "[$(date '+%F %T')] === w=0.5（89.01 底座）开始 ===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py \
    --config configs/baseline_config.yaml --set \
    DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16 \
    TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 \
    TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 \
    TRAIN.cam_consistency=0.5 TRAIN.cam_consistency_T=2.0 \
    TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 \
    TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 \
    TRAIN.backbone_lr_layers.layer4=0.05 \
    MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k \
    "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]" \
    EXP_NAME=swsl_448_best_w05 >> "$LOG" 2>&1
echo "[$(date '+%F %T')] === w=0.5 结束 ===" >> "$LOG"

echo "[$(date '+%F %T')] === H: w=0.8（88.62 底座）开始 ===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py \
    --config configs/baseline_config.yaml --set \
    DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16 \
    TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=1.0 \
    TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 \
    TRAIN.cam_consistency=0.8 TRAIN.cam_consistency_T=2.0 \
    TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 \
    TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 \
    TRAIN.backbone_lr_layers.layer4=0.1 \
    MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k \
    "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]" \
    EXP_NAME=swsl_448_cons_w08 >> "$LOG" 2>&1
echo "[$(date '+%F %T')] === 全部完成 ===" >> "$LOG"
