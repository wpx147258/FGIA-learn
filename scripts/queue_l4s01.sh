#!/usr/bin/env bash
# 等 alpha02_p07ls 跑完后，起「layer4 学习率减半」那一轮。
#
# 分层 lr 现在是 stem/l1 0.01、l2 0.03、l3 0.05、l4 0.2（×1e-3，即 l4=2e-4）。
# 这一轮把 layer4 降到 0.1（=1e-4），其余不动。
#
# 目的：把分层的收益拆开——
#   固定 0.1（全 backbone 1e-4）              87.07
#   分层，l4=2e-4                            87.50
#   分层，l4=1e-4                            ← 这一轮
# 如果 l4=1e-4 更好，说明收益主要来自「压住低层」；如果更差，说明 l4 抬高也在起作用。
# cam_alpha_crop 取 0.3，和 swsl_448_layers（87.50）一致，保证只差 l4 一个变量。
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/home/wpx/anaconda3/bin/python
TARGET='EXP_NAME=swsl_448_alpha02_p07ls'
LOG=outputs/_logs/queue_l4s01.log

echo "[$(date '+%F %T')] 阶段1: 等 p07ls 启动 ..." >> "$LOG"
until pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done

echo "[$(date '+%F %T')] 阶段2: p07ls 已启动，等它跑完 ..." >> "$LOG"
while pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done

echo "[$(date '+%F %T')] GPU 空闲，启动 layer4=0.1" >> "$LOG"
setsid nohup systemd-inhibit --what=sleep --mode=block "$PY" train.py \
    --config configs/baseline_config.yaml --set \
    DATASET.image_size=448 \
    TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.epochs=50 TRAIN.batch_size=16 \
    TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.3 TRAIN.cam_margin=1.3 \
    TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 \
    TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 \
    TRAIN.backbone_lr_layers.layer4=0.1 \
    "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]" \
    MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k \
    EXP_NAME=swsl_448_layers_l4s01 >> "$LOG" 2>&1 < /dev/null &

echo "[$(date '+%F %T')] 已启动 pid=$!" >> "$LOG"
