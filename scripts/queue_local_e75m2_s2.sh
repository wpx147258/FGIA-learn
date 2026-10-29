#!/usr/bin/env bash
# 本机：75 轮 + 双尺度一致性，**SEED=1**。
#
# 为什么要这一格：本地 SEED=7 上 e75m2_s7 对同机同种子基线是 **+0.62（p=0.02，显著）**，
# 但远程 SEED=42 的同配置只有 +0.09（p=0.78）。观察到的规律是
# "e75m2 在两台机器两个种子上都落在 89.68，而基线在 89.07~89.94 之间跳"——
# 更像是**方差压缩**而不是增益。本地已有 base_s1=89.26，
# 这一格给出第二对同机同种子对照，直接判定上面两种解释哪个成立。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/queue_r50_90.log
echo "[$(date '+%F %T')] queue_local_e75m2_s2 等待 base_s2 结束 ..." >> "$LOG"
until grep -qs "queue_local_base_s2 结束" "$LOG"; do sleep 30; done
BASE=(
  --config configs/baseline_config.yaml --set
  DATASET.image_size=448 TRAIN.epochs=75 TRAIN.batch_size=16
  TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05
  TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3
  TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0
  TRAIN.cam_margin2=1.6 TRAIN.cam_consistency2=0.15
  TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01
  TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05
  TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3
  MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1
  "TRAIN.swa_start_frac=[0.6, 0.7, 0.8]" DATASET.train_on_val=true
)
name=r50_all_e75m2_s2
echo "[$(date '+%F %T')] START $name" >> "$LOG"
"$PY" train.py "${BASE[@]}" SEED=2 EXP_NAME="$name" > "logs/${name}_train.log" 2>&1
d=$(ls -dt "outputs/${name}"/*/ 2>/dev/null | head -1)
echo "[$(date '+%F %T')] DONE $name -> $d" >> "$LOG"
"$PY" scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] queue_local_e75m2_s2 结束" >> "$LOG"
