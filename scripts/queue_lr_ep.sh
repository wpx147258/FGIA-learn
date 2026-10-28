#!/usr/bin/env bash
# 正则族四个变量都没有降低记忆（四个 run 的干净训练集 top1 全是 100.00%），
# 所以下一步换 B/C 类：头学习率和训练长度。
#
#   H1: 头 lr 1e-3 → 5e-4      （更慢的拟合 = 隐式正则；也可能欠拟合）
#   H2: 头 lr 1e-3 → 2e-3      （更快，大概率更差，用来定方向）
#   H3: epochs 50→70 + lr×5/7  （保持 lr×epochs 不变，骨干偏离不增大）
#
# 底座 = swsl_448_cons_l4_p07ls（test 89.01）。
# 注意 backbone 的 lr = 头 lr × 分层倍率，所以改 TRAIN.lr 会连同 backbone 一起缩放——
# H1/H2 实际上是「整体学习率」的扫描，不只是头。这点在解读时要记住。
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/home/wpx/anaconda3/bin/python
LOG=outputs/_logs/queue_lr_ep.log
BASE=(
  --config configs/baseline_config.yaml --set
  DATASET.image_size=448 TRAIN.batch_size=16
  TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05
  TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3
  TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0
  TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01
  TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05
  TRAIN.backbone_lr_layers.layer4=0.05
  MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k
  "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]"
)

echo "[$(date '+%F %T')] === H1: lr=5e-4 ===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py "${BASE[@]}" \
    TRAIN.epochs=50 TRAIN.lr=5e-4 EXP_NAME=swsl_448_best_lr5e4 >> "$LOG" 2>&1

echo "[$(date '+%F %T')] === H2: lr=2e-3 ===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py "${BASE[@]}" \
    TRAIN.epochs=50 TRAIN.lr=2e-3 EXP_NAME=swsl_448_best_lr2e3 >> "$LOG" 2>&1

echo "[$(date '+%F %T')] === H3: epochs=70, lr=7.14e-4 ===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py "${BASE[@]}" \
    TRAIN.epochs=70 TRAIN.lr=7.14e-4 EXP_NAME=swsl_448_best_e70 >> "$LOG" 2>&1

echo "[$(date '+%F %T')] === 全部完成 ===" >> "$LOG"
