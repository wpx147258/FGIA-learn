#!/usr/bin/env bash
# 等 min_lr=2e-6 那轮**跑完**之后，再起「CAM 裁剪框对齐」那一轮。
#
# ⚠️ 必须两段等待：mlr2e6 这一轮此刻还没启动（它自己在等 nocutmix 结束），
#    如果只写 `while pgrep mlr2e6; do sleep; done`，循环会立刻退出，
#    alpha 就会和 nocutmix 抢同一块卡。
#
# 背景（实测）：SWSL 的 CAM 比 a1 集中得多，同样 alpha=0.3 时
#   掩码覆盖 3.1% vs 7.2%，实际裁剪面积 16.0% vs 25.5%（中位 10.3% vs 21.1%）
# 这一轮把 cam_alpha_crop 从 0.3 降到 0.2（面积 21.9%，接近 a1 的 25.5%），
# 其余配置和 swsl_448_layers（test 87.50）逐项相同。
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/home/wpx/anaconda3/bin/python
TARGET='EXP_NAME=swsl_448_layers_mlr2e6'
LOG=outputs/_logs/queue_alpha02.log

echo "[$(date '+%F %T')] 阶段1: 等 mlr2e6 启动 ..." >> "$LOG"
until pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done

echo "[$(date '+%F %T')] 阶段2: mlr2e6 已启动，等它跑完 ..." >> "$LOG"
while pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done

echo "[$(date '+%F %T')] GPU 空闲，启动 alpha=0.2" >> "$LOG"
setsid nohup systemd-inhibit --what=sleep --mode=block "$PY" train.py \
    --config configs/baseline_config.yaml --set \
    DATASET.image_size=448 \
    TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.epochs=50 TRAIN.batch_size=16 \
    TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 \
    TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 \
    TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 \
    TRAIN.backbone_lr_layers.layer4=0.2 \
    "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]" \
    MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k \
    EXP_NAME=swsl_448_alpha02 >> "$LOG" 2>&1 < /dev/null &

echo "[$(date '+%F %T')] 已启动 pid=$!" >> "$LOG"
