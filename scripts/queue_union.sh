#!/usr/bin/env bash
# 等 l4s01 跑完后，起「union 裁剪 + mixup_p 0.7」那一轮。
#
# 依据（实测 48~96 个混合样本）：
#   mixup batch 上 A 的 CAM 框 14.3%、B 的 9.4%，两者 IoU 仅 0.037
#   —— 只裁 A 的框时 patch 里没有 B 的证据，却要求它输出 A+B 的软目标，监督矛盾。
#   取并集后：union 框 31.5%。
#
# 和 mixup_p=0.7 组合后的平均裁剪尺度：
#   30% 干净样本 -> 12%   70% mixup -> 31.5%   平均 25.7%
#   恰好等于 a1 那版（25.5%），同时保留紧/松两种尺度。
# margin 用 1.1（不是 1.3，也不是 1.0）：union 框本来就大，1.3 会外扩到 53%；
# 1.1 保留一点上下文，平均裁剪面积 ~31%（干净 14.5% / union 38%）。
#
# 对照链：alpha02(p1.0) -> p07ls(p0.7) -> 本轮(p0.7 + union)
#   前者分离 mixup_p 的作用，本轮分离 union 的作用。
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/home/wpx/anaconda3/bin/python
TARGET='EXP_NAME=swsl_448_layers_l4s01'
LOG=outputs/_logs/queue_union.log

echo "[$(date '+%F %T')] 阶段1: 等 l4s01 启动 ..." >> "$LOG"
until pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done

echo "[$(date '+%F %T')] 阶段2: l4s01 已启动，等它跑完 ..." >> "$LOG"
while pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done

echo "[$(date '+%F %T')] GPU 空闲，启动 union + p0.7" >> "$LOG"
setsid nohup systemd-inhibit --what=sleep --mode=block "$PY" train.py \
    --config configs/baseline_config.yaml --set \
    DATASET.image_size=448 \
    TRAIN.mixup=0.2 TRAIN.cutmix=0.0 TRAIN.mixup_p=0.7 \
    TRAIN.epochs=50 TRAIN.batch_size=16 \
    TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 TRAIN.cam_margin_union=1.1 \
    TRAIN.cam_crop_union=true \
    TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 \
    TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 \
    TRAIN.backbone_lr_layers.layer4=0.2 \
    "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]" \
    MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k \
    EXP_NAME=swsl_448_p07_union >> "$LOG" 2>&1 < /dev/null &

echo "[$(date '+%F %T')] 已启动 pid=$!" >> "$LOG"
