#!/usr/bin/env bash
# 重排 p07+ls（上一次因为输出目录被外部移走而崩在第一个 epoch，
# 见 outputs/swsl_448_alpha02_p07ls/20261021-003728/log.txt）。
#
# 这一轮排在 union 之后，配置与上次相同：
#   cutmix 0.0 + mixup 0.2 + mixup_p 0.7 + label_smoothing 0.05
#   cam_alpha_crop 0.2（和 alpha02 一致，两者只差 mixup/ls 一个变量）
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/home/wpx/anaconda3/bin/python
TARGET='EXP_NAME=swsl_448_p07_union'
LOG=outputs/_logs/queue_p07ls_v2.log

echo "[$(date '+%F %T')] 阶段1: 等 union 那轮启动 ..." >> "$LOG"
until pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done

echo "[$(date '+%F %T')] 阶段2: union 已启动，等它跑完 ..." >> "$LOG"
while pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done

echo "[$(date '+%F %T')] GPU 空闲，启动 p0.7 + ls0.05（重跑）" >> "$LOG"
setsid nohup systemd-inhibit --what=sleep --mode=block "$PY" train.py \
    --config configs/baseline_config.yaml --set \
    DATASET.image_size=448 \
    TRAIN.mixup=0.2 TRAIN.cutmix=0.0 TRAIN.mixup_p=0.7 \
    TRAIN.label_smoothing=0.05 \
    TRAIN.epochs=50 TRAIN.batch_size=16 \
    TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 \
    TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 \
    TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 \
    TRAIN.backbone_lr_layers.layer4=0.2 \
    "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]" \
    MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k \
    EXP_NAME=swsl_448_p07ls_v2 >> "$LOG" 2>&1 < /dev/null &

echo "[$(date '+%F %T')] 已启动 pid=$!" >> "$LOG"
