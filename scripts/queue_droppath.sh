#!/usr/bin/env bash
# 等 p07ls_v2 跑完后，起「随机深度」那一轮。
#
# 依据：当前模型在**干净训练图上 top1 = 100.00%**（5395 张全对），测试只有 87.6%
# —— 完全是记忆而非泛化。最混淆的 12 对类（燕鸥/乌鸦/鸬鹚/鸥）训练全对、测试 20~50%，
# 特征中心余弦 0.89~0.93。
# 这种情况下再加容量/注意力没用，该做的是抑制记忆。drop_path 一直没开过（0.0）。
#
# 基线配置 = 当前 SWSL 最好的那版（分层 lr + alpha 0.2，test 87.65），只加 drop_path。
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/home/wpx/anaconda3/bin/python
TARGET='EXP_NAME=swsl_448_p07ls_v2'
LOG=outputs/_logs/queue_droppath.log

echo "[$(date '+%F %T')] 阶段1: 等 p07ls_v2 启动 ..." >> "$LOG"
until pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done

echo "[$(date '+%F %T')] 阶段2: p07ls_v2 已启动，等它跑完 ..." >> "$LOG"
while pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done

echo "[$(date '+%F %T')] GPU 空闲，启动 drop_path=0.1" >> "$LOG"
setsid nohup systemd-inhibit --what=sleep --mode=block "$PY" train.py \
    --config configs/baseline_config.yaml --set \
    DATASET.image_size=448 \
    TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.epochs=50 TRAIN.batch_size=16 \
    TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 \
    TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 \
    TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 \
    TRAIN.backbone_lr_layers.layer4=0.2 \
    MODEL.kwargs.drop_path_rate=0.1 \
    MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k \
    "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]" \
    EXP_NAME=swsl_448_dp01 >> "$LOG" 2>&1 < /dev/null &

echo "[$(date '+%F %T')] 已启动 pid=$!" >> "$LOG"
