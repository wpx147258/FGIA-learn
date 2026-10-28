#!/usr/bin/env bash
# 重跑 H1/H2（上一版崩了：TRAIN.lr=5e-4 被 YAML 解析成字符串，lr×倍率时 TypeError）。
# 正确写法必须带小数点：5.0e-4 / 2.0e-3。
#
#   H1: 整体 lr 1e-3 → 5.0e-4
#   H2: 整体 lr 1e-3 → 2.0e-3
# 底座 = swsl_448_cons_l4_p07ls（test 89.01）。
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/home/wpx/anaconda3/bin/python
TARGET='EXP_NAME=swsl_448_best_e70'
LOG=outputs/_logs/queue_lr2.log
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

echo "[$(date '+%F %T')] 阶段1: 等 e70 启动 ..." >> "$LOG"
until pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done
echo "[$(date '+%F %T')] 阶段2: e70 已启动，等它跑完 ..." >> "$LOG"
while pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done

echo "[$(date '+%F %T')] === H1: lr=5.0e-4 ===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py "${BASE[@]}" \
    TRAIN.lr=5.0e-4 EXP_NAME=swsl_448_best_lr5e4 >> "$LOG" 2>&1

echo "[$(date '+%F %T')] === H2: lr=2.0e-3 ===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py "${BASE[@]}" \
    TRAIN.lr=2.0e-3 EXP_NAME=swsl_448_best_lr2e3 >> "$LOG" 2>&1

echo "[$(date '+%F %T')] === 全部完成 ===" >> "$LOG"
