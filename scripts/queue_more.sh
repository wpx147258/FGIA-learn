#!/usr/bin/env bash
# 等 B（cons + l4=0.05 + p0.7/ls）跑完后，继续跑：
#   C: 一致性权重 0.5   —— 0.3 是拍的，扫一下上下界
#   D: 一致性权重 0.15
#   E: 一致性 + p0.7/ls（layer4 保持 0.1）—— 在已验证的 cons 底座上叠第二个改动
#
# C/D 的底座 = swsl_448_cons（test 88.62）：l4=0.1、cutmix1.0、p1.0、alpha0.2。
# 这样它们和 88.62 只差一个权重，是干净的单变量对照。
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/home/wpx/anaconda3/bin/python
TARGET='EXP_NAME=swsl_448_cons_l4_p07ls'
LOG=outputs/_logs/queue_more.log
BASE=(
  --config configs/baseline_config.yaml --set
  DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16
  TRAIN.cutmix=1.0 TRAIN.mixup=0.2
  TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3
  TRAIN.cam_consistency_T=2.0
  TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01
  TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05
  TRAIN.backbone_lr_layers.layer4=0.1
  MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k
  "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]"
)

echo "[$(date '+%F %T')] 阶段1: 等 B 启动 ..." >> "$LOG"
until pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done
echo "[$(date '+%F %T')] 阶段2: B 已启动，等它跑完 ..." >> "$LOG"
while pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done

echo "[$(date '+%F %T')] === C: cons w=0.5 ===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py "${BASE[@]}" \
    TRAIN.cam_consistency=0.5 EXP_NAME=swsl_448_cons_w05 >> "$LOG" 2>&1

echo "[$(date '+%F %T')] === D: cons w=0.15 ===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py "${BASE[@]}" \
    TRAIN.cam_consistency=0.15 EXP_NAME=swsl_448_cons_w015 >> "$LOG" 2>&1

echo "[$(date '+%F %T')] === E: cons + p0.7/ls ===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py "${BASE[@]}" \
    TRAIN.cam_consistency=0.3 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 \
    EXP_NAME=swsl_448_cons_p07ls >> "$LOG" 2>&1

echo "[$(date '+%F %T')] === 全部完成 ===" >> "$LOG"
