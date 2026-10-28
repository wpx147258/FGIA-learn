#!/usr/bin/env bash
# 等主类 CAM 那轮跑完后，起「quad 换掉一部分 cutmix」。
#
# 设计（按你的要求）：把 cutmix 从 35% 降到 20%，空出来的那部分给 quad。
#   mixup_p=0.7 + switch_p=0.714  ->  clean 30% / mixup 50% / cutmix 20%
#   cam_quad=0.2（损失权重）+ cam_quad_p=0.25（逐 batch 触发）
#   -> quad 实际作用在约 0.70×0.25 ≈ 15% 的 batch 上
#
# 底座 = swsl_448_cons_l4_p07ls（test 89.01），一致性/crop/lr 全部保留。
# quad 是辅助分支（打乱 2×2 后再前向一次），不裁剪，不加参数。
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/home/wpx/anaconda3/bin/python
TARGET='EXP_NAME=swsl_448_best_majority'
LOG=outputs/_logs/queue_quad.log

echo "[$(date '+%F %T')] 阶段1: 等主类 CAM 那轮启动 ..." >> "$LOG"
until pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done
echo "[$(date '+%F %T')] 阶段2: 已启动，等它跑完 ..." >> "$LOG"
while pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done

echo "[$(date '+%F %T')] === quad 轮开始 ===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py \
    --config configs/baseline_config.yaml --set \
    DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16 \
    TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.switch_p=0.714 \
    TRAIN.label_smoothing=0.05 \
    TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 \
    TRAIN.cam_quad=0.2 TRAIN.cam_quad_p=0.25 \
    TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 \
    TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 \
    TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 \
    TRAIN.backbone_lr_layers.layer4=0.05 \
    MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k \
    "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]" \
    EXP_NAME=swsl_448_best_quad >> "$LOG" 2>&1

echo "[$(date '+%F %T')] === 完成 ===" >> "$LOG"
