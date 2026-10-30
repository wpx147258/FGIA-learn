#!/usr/bin/env bash
# 本机：**只加双尺度一致性，不加训练长度**（50 轮 + margin2 1.6 / cons2 0.15），SEED=1。
#
# 归因问题：`e75m2` 一直捆绑了两个改动（50→75 轮、单尺度→双尺度一致性），
# 而远程的拆解显示 e75（只加长度）是 89.51（−0.08），e75_m2 是 89.68（+0.09），
# 说明**起作用的可能是双尺度这个损失项，而不是训练长度**。
# 本地 e75m2 两粒是 89.58 / 89.68，对本地基线均值 89.27 是 +0.3~0.4。
#
# 这一格把长度拿掉：若 50 轮 + 双尺度 也能到 89.6 左右，则
#   (a) 归因确认是 m2，(b) 配方便宜三分之一，(c) 可以叠加到更长的训练上再试。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/queue_r50_90.log
echo "[$(date '+%F %T')] queue_local_m2_s1 等待 e75m2_s2 结束 ..." >> "$LOG"
until grep -qs "queue_local_e75m2_s2 结束" "$LOG"; do sleep 30; done
BASE=(
  --config configs/baseline_config.yaml --set
  DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16
  TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05
  TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3
  TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0
  TRAIN.cam_margin2=1.6 TRAIN.cam_consistency2=0.15
  TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01
  TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05
  TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3
  MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1
  "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]" DATASET.train_on_val=true
)
name=r50_all_m2_s1
echo "[$(date '+%F %T')] START $name" >> "$LOG"
"$PY" train.py "${BASE[@]}" SEED=1 EXP_NAME="$name" > "logs/${name}_train.log" 2>&1
d=$(ls -dt "outputs/${name}"/*/ 2>/dev/null | head -1)
echo "[$(date '+%F %T')] DONE $name -> $d" >> "$LOG"
"$PY" scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] queue_local_m2_s1 结束" >> "$LOG"
