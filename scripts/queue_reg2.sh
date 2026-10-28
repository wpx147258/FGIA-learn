#!/usr/bin/env bash
# 等 dropout 那轮跑完后，再测两个正则变量（用户的 A 类清单）。
#
#   G1: label_smoothing 0.05 → 0.1   （只在干净 batch 上生效）
#   G2: batch_size 16 → 32           （影响 BN 统计与梯度噪声；lr 不动）
#
# 底座 = swsl_448_cons_l4_p07ls（test 89.01），只改一个变量。
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/home/wpx/anaconda3/bin/python
TARGET='EXP_NAME=swsl_448_best_drop10'
LOG=outputs/_logs/queue_reg2.log
BASE=(
  --config configs/baseline_config.yaml --set
  DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16
  TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05
  TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3
  TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0
  TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01
  TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05
  TRAIN.backbone_lr_layers.layer4=0.05
  MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k
  "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]"
)

echo "[$(date '+%F %T')] 阶段1: 等 dropout 那轮启动 ..." >> "$LOG"
until pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done
echo "[$(date '+%F %T')] 阶段2: 已启动，等它跑完 ..." >> "$LOG"
while pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done

echo "[$(date '+%F %T')] === G1: label_smoothing=0.1 ===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py "${BASE[@]}" \
    TRAIN.label_smoothing=0.1 EXP_NAME=swsl_448_best_ls10 >> "$LOG" 2>&1

echo "[$(date '+%F %T')] === G2: batch_size=32 ===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py "${BASE[@]}" \
    TRAIN.batch_size=32 EXP_NAME=swsl_448_best_bs32 >> "$LOG" 2>&1

echo "[$(date '+%F %T')] === 全部完成 ===" >> "$LOG"
