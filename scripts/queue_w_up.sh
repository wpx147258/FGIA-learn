#!/usr/bin/env bash
# 一致性权重往 0.5 以上推，并把最好的组合（89.01）作为新底座。
#
# 已实测的权重趋势（底座 l4=0.1 / cutmix1.0 / p1.0）：
#   w=0.15 → 88.46    w=0.3 → 88.62    w=0.5 → 88.80     ← 单调上升
# 最好的组合（89.01）= 一致性 w0.3 + l4=0.05 + p0.7/ls。
#
# F: w=0.8 + l4=0.05 + p0.7/ls   （直接在 89.01 上抬权重）
# G: w=1.2 + l4=0.05 + p0.7/ls   （探上界）
# H: w=0.8，其余回到 88.62 的底座（确认趋势不是只在某个配置里成立）
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/home/wpx/anaconda3/bin/python
LOG=outputs/_logs/queue_w_up.log
COMMON=(
  --config configs/baseline_config.yaml --set
  DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16
  TRAIN.mixup=0.2 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3
  TRAIN.cam_consistency_T=2.0
  TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01
  TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05
  MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k
  "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]"
)

echo "[$(date '+%F %T')] === F: w=0.8 + l4=0.05 + p0.7/ls ===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py "${COMMON[@]}" \
    TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 \
    TRAIN.cam_consistency=0.8 TRAIN.backbone_lr_layers.layer4=0.05 \
    EXP_NAME=swsl_448_best_w08 >> "$LOG" 2>&1

echo "[$(date '+%F %T')] === G: w=1.2 + l4=0.05 + p0.7/ls ===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py "${COMMON[@]}" \
    TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 \
    TRAIN.cam_consistency=1.2 TRAIN.backbone_lr_layers.layer4=0.05 \
    EXP_NAME=swsl_448_best_w12 >> "$LOG" 2>&1

echo "[$(date '+%F %T')] === H: w=0.8（88.62 底座）===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py "${COMMON[@]}" \
    TRAIN.cutmix=1.0 TRAIN.mixup_p=1.0 \
    TRAIN.cam_consistency=0.8 TRAIN.backbone_lr_layers.layer4=0.1 \
    EXP_NAME=swsl_448_cons_w08 >> "$LOG" 2>&1

echo "[$(date '+%F %T')] === 全部完成 ===" >> "$LOG"
