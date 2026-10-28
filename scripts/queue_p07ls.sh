#!/usr/bin/env bash
# 等 alpha02 跑完后，起「p0.7 + label smoothing」那一轮。
#
# 依据：
#   a1 上  无 cutmix + 全 mixup (p=1.0, ls=0)     86.68
#          无 cutmix + p=0.7 + ls=0.05            87.11   (+0.43)
#   而 SWSL 上「无 cutmix」只比「有 cutmix」低 0.14，代价很小，
#   所以「干净样本 + 轻度 ls」有机会净赚。
#
# alpha_crop 取 0.2，和 alpha02 一致 —— 这样 p07ls 与 alpha02 之间只差 mixup/ls 一个变量。
# 注意 p=0.7 意味着 30% 的 batch 没有 soft_target，走普通 CE，label_smoothing 才生效。
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/home/wpx/anaconda3/bin/python
TARGET='EXP_NAME=swsl_448_alpha02'
LOG=outputs/_logs/queue_p07ls.log

echo "[$(date '+%F %T')] 阶段1: 等 alpha02 启动 ..." >> "$LOG"
until pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done

echo "[$(date '+%F %T')] 阶段2: alpha02 已启动，等它跑完 ..." >> "$LOG"
while pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done

echo "[$(date '+%F %T')] GPU 空闲，启动 p0.7 + ls0.05" >> "$LOG"
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
    EXP_NAME=swsl_448_alpha02_p07ls >> "$LOG" 2>&1 < /dev/null &

echo "[$(date '+%F %T')] 已启动 pid=$!" >> "$LOG"
