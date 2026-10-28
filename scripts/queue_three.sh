#!/usr/bin/env bash
# 等 l4s005 跑完后，串行跑三个新方法（每个约 50 分钟，共 ~2.5 小时）。
#
#   ① CBP 二阶池化   —— SWSL 的特征秩 182（a1 只有 110），协方差的独立结构翻倍
#   ② 裁剪一致性     —— 主分支与裁剪分支对称 KL，逼主干从「能被放大的局部」取信息
#   ③ manifold mixup —— 特征层插值，惩罚类间非线性空隙（最难那 14 个类簇重叠）
#
# 三个都以当前最强的 SWSL 配置为底：分层 lr（layer4=0.1，实测 87.83）+ alpha 0.2。
# 训练在前台跑，脚本本身用 setsid 脱离终端，所以会话结束也不会中断。
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/home/wpx/anaconda3/bin/python
TARGET='EXP_NAME=swsl_448_l4s005'
LOG=outputs/_logs/queue_three.log
COMMON=(
  --config configs/baseline_config.yaml --set
  DATASET.image_size=448
  TRAIN.epochs=50 TRAIN.batch_size=16
  TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3
  TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01
  TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05
  TRAIN.backbone_lr_layers.layer4=0.1
  MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k
)

echo "[$(date '+%F %T')] 阶段1: 等 l4s005 启动 ..." >> "$LOG"
until pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done
echo "[$(date '+%F %T')] 阶段2: l4s005 已启动，等它跑完 ..." >> "$LOG"
while pgrep -f "$TARGET" >/dev/null 2>&1; do sleep 15; done

echo "[$(date '+%F %T')] === ① CBP 开始 ===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py "${COMMON[@]}" \
    TRAIN.mixup=0.2 TRAIN.cutmix=1.0 MODEL.head=cbp \
    "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]" EXP_NAME=swsl_448_cbp \
    >> "$LOG" 2>&1
echo "[$(date '+%F %T')] === ① CBP 结束 ===" >> "$LOG"

echo "[$(date '+%F %T')] === ② 裁剪一致性开始 ===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py "${COMMON[@]}" \
    TRAIN.mixup=0.2 TRAIN.cutmix=1.0 MODEL.head=linear \
    TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 \
    "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]" EXP_NAME=swsl_448_cons \
    >> "$LOG" 2>&1
echo "[$(date '+%F %T')] === ② 结束 ===" >> "$LOG"

echo "[$(date '+%F %T')] === ③ manifold mixup 开始 ===" >> "$LOG"
systemd-inhibit --what=sleep --mode=block "$PY" train.py "${COMMON[@]}" \
    TRAIN.mixup=0.0 TRAIN.cutmix=0.0 TRAIN.manifold_mixup=0.4 MODEL.head=linear \
    "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]" EXP_NAME=swsl_448_mmix \
    >> "$LOG" 2>&1
echo "[$(date '+%F %T')] === ③ 结束，全部完成 ===" >> "$LOG"
