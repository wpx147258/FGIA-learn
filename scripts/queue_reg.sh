#!/usr/bin/env bash
# 等 quad 轮跑完后，测两个正则变量（诊断：训练集 top1=100%，记忆明显）。
#
#   F1: weight_decay 0.05 → 0.1     （从没测过）
#   F2: MODEL.dropout 0.0 → 0.1     （头 410K，作用温和）
#
# 底座 = swsl_448_cons_l4_p07ls（test 89.01），只改一个变量。
# 注意：这个底座已经有三个正则（p0.7/ls、一致性、低 l4），加正则很可能过量——
# 所以要同时看「是否掉点」，掉点就说明 89.01 那个组合已经接近正则上限。
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/home/wpx/anaconda3/bin/python
TARGET='EXP_NAME=swsl_448_best_quad'
LOG=outputs/_logs/queue_reg.log
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

echo "[$(date '+%F %T')] 阶段1: 等 quad 启动 ..." >> "$LOG"
until pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done
echo "[$(date '+%F %T')] 阶段2: quad 已启动，等它跑完 ..." >> "$LOG"
while pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done

echo "[$(date '+%F %T')] === F1: weight_decay=0.1 ===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py "${BASE[@]}" \
    TRAIN.weight_decay=0.1 EXP_NAME=swsl_448_best_wd10 >> "$LOG" 2>&1

echo "[$(date '+%F %T')] === F2: dropout=0.1 ===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py "${BASE[@]}" \
    MODEL.dropout=0.1 EXP_NAME=swsl_448_best_drop10 >> "$LOG" 2>&1

echo "[$(date '+%F %T')] === 全部完成 ===" >> "$LOG"
