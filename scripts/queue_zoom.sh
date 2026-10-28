#!/usr/bin/env bash
# 等 drop_path 那轮跑完后，起「训练时更多放大」那一轮。
#
# 依据：诊断显示目标框占图面积最小的五分之一准确率只有 84.47%，
# 最大的那批 89.56%（相差 5 个点）。当前训练用 RandomResizedCrop(0.5, 1.0)，
# 平均只放大 1.15 倍；下限放到 0.35 后线性放大最多 1.69 倍。
#
# ⚠️ 只改训练增强，评测完全不动（评估走 Resize(448/crop_pct)+CenterCrop）。
# 基线配置 = 当前 SWSL 最好的那版（分层 lr + alpha 0.2，test 87.65）。
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/home/wpx/anaconda3/bin/python
TARGET='EXP_NAME=swsl_448_dp01'
LOG=outputs/_logs/queue_zoom.log

echo "[$(date '+%F %T')] 阶段1: 等 dp01 启动 ..." >> "$LOG"
until pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done

echo "[$(date '+%F %T')] 阶段2: dp01 已启动，等它跑完 ..." >> "$LOG"
while pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done

echo "[$(date '+%F %T')] GPU 空闲，启动 crop_scale=0.35" >> "$LOG"
setsid nohup systemd-inhibit --what=sleep --mode=block "$PY" train.py \
    --config configs/baseline_config.yaml --set \
    DATASET.image_size=448 \
    TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.epochs=50 TRAIN.batch_size=16 \
    "TRAIN.crop_scale=[0.35, 1.0]" \
    TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 \
    TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 \
    TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 \
    TRAIN.backbone_lr_layers.layer4=0.2 \
    MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k \
    "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]" \
    EXP_NAME=swsl_448_zoom035 >> "$LOG" 2>&1 < /dev/null &

echo "[$(date '+%F %T')] 已启动 pid=$!" >> "$LOG"
