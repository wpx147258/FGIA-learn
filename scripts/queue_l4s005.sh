#!/usr/bin/env bash
# 等 drop_path 那轮跑完后，起「layer4 lr = 0.05」那一轮。
#
# 依据（骨干偏离 vs 准确率，实测三点）：
#   layer4 lr 2e-4  →  相对偏离 1.336  →  test 87.65
#   layer4 lr 1e-4  →  相对偏离 0.748  →  test 87.83
#   a1（1e-3）      →  相对偏离 0.337  →  test 87.84
# 偏离和 lr 近似线性，要降到 a1 那个 0.337 需要 l4≈4.5e-5。
# 这一轮取 0.05（=5e-5），预期偏离落在 0.35~0.40。
#
# 其余配置和 swsl_448_layers_l4s01（test 87.83）逐项相同，只改 layer4 这一个数，
# 保证是单变量对照。
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/home/wpx/anaconda3/bin/python
TARGET='EXP_NAME=swsl_448_dp01'
LOG=outputs/_logs/queue_l4s005.log

echo "[$(date '+%F %T')] 阶段1: 等 dp01 启动 ..." >> "$LOG"
until pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done

echo "[$(date '+%F %T')] 阶段2: dp01 已启动，等它跑完 ..." >> "$LOG"
while pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done

echo "[$(date '+%F %T')] GPU 空闲，启动 layer4=0.05" >> "$LOG"
setsid nohup systemd-inhibit --what=sleep --mode=block "$PY" train.py \
    --config configs/baseline_config.yaml --set \
    DATASET.image_size=448 \
    TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.epochs=50 TRAIN.batch_size=16 \
    TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.3 TRAIN.cam_margin=1.3 \
    TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 \
    TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 \
    TRAIN.backbone_lr_layers.layer4=0.05 \
    MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k \
    "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]" \
    EXP_NAME=swsl_448_l4s005 >> "$LOG" 2>&1 < /dev/null &

echo "[$(date '+%F %T')] 已启动 pid=$!" >> "$LOG"
