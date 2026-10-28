#!/usr/bin/env bash
# 把昨晚三个独立有效的改动叠起来。
#
# 单变量结果（都以 l4=0.1 + cutmix1.0 + p1.0 为底，test 88.62 是 cons 那轮）：
#   裁剪一致性 w=0.3        88.62   (+0.79)
#   layer4 lr 0.1→0.05      88.24   (+0.41)
#   p0.7 + ls0.05（无cutmix）88.34   (+0.51)
#
# A = 一致性 + l4=0.05
# B = A + p0.7/ls（cutmix 保留）
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/home/wpx/anaconda3/bin/python
LOG=outputs/_logs/queue_stack.log
COMMON=(
  --config configs/baseline_config.yaml --set
  DATASET.image_size=448
  TRAIN.epochs=50 TRAIN.batch_size=16
  TRAIN.cutmix=1.0 TRAIN.mixup=0.2
  TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3
  TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0
  TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01
  TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05
  TRAIN.backbone_lr_layers.layer4=0.05
  MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k
  "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]"
)

echo "[$(date '+%F %T')] === A: 一致性 + l4=0.05 开始 ===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py "${COMMON[@]}" \
    EXP_NAME=swsl_448_cons_l4s005 >> "$LOG" 2>&1
echo "[$(date '+%F %T')] === A 结束 ===" >> "$LOG"

echo "[$(date '+%F %T')] === B: A + p0.7/ls 开始 ===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py "${COMMON[@]}" \
    TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 \
    EXP_NAME=swsl_448_cons_l4_p07ls >> "$LOG" 2>&1
echo "[$(date '+%F %T')] === B 结束，全部完成 ===" >> "$LOG"
