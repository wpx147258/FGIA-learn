#!/usr/bin/env bash
# 等 H（cons w=0.8）跑完后，在**当前最优配置**上测「用主类算 CAM」。
#
# 最优配置 = swsl_448_cons_l4_p07ls（test 89.01）：
#   一致性 w=0.3 + layer4 lr 0.05 + cutmix1.0 + p0.7 + ls0.05
# 只加 cam_crop_majority=true 一个变量。
#
# 影响面：cutmix 的 batch 本来就不走 CAM，所以只影响 mixup 那一批里
# A 权重 <0.5 的样本（约 51% 的 mixup 样本）。
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/home/wpx/anaconda3/bin/python
TARGET='EXP_NAME=swsl_448_cons_w08'
LOG=outputs/_logs/queue_majority.log

echo "[$(date '+%F %T')] 阶段1: 等 H 启动 ..." >> "$LOG"
until pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done
echo "[$(date '+%F %T')] 阶段2: H 已启动，等它跑完 ..." >> "$LOG"
while pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done

echo "[$(date '+%F %T')] === 主类 CAM 开始 ===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py \
    --config configs/baseline_config.yaml --set \
    DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16 \
    TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 \
    TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 \
    TRAIN.cam_crop_majority=true \
    TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 \
    TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 \
    TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 \
    TRAIN.backbone_lr_layers.layer4=0.05 \
    MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k \
    "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]" \
    EXP_NAME=swsl_448_best_majority >> "$LOG" 2>&1

echo "[$(date '+%F %T')] === 完成 ===" >> "$LOG"
