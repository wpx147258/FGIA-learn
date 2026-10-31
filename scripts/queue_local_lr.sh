#!/usr/bin/env bash
# 2026-10-31：**分层学习率**——backbone_lr_layers 的 0.01/0.01/0.03/0.05/0.05 是项目第一天拍的值，
# 全训口径下从未扫过。这一格换成**统一 0.1**（backbone 全部 1e-4，标准微调学习率）。
# 基于 e75m2，SEED=42。参照：克隆上 e75m2_s42r = 89.68（同配方、分层 LR）。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/queue_r50_90.log
BASE=(
  --config configs/baseline_config.yaml --set
  DATASET.image_size=448 TRAIN.epochs=75 TRAIN.batch_size=16
  TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05
  TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3
  TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0
  TRAIN.cam_margin2=1.6 TRAIN.cam_consistency2=0.15
  TRAIN.backbone_lr_layers.stem=0.1 TRAIN.backbone_lr_layers.layer1=0.1
  TRAIN.backbone_lr_layers.layer2=0.1 TRAIN.backbone_lr_layers.layer3=0.1
  TRAIN.backbone_lr_layers.layer4=0.1
  TRAIN.lr=1.0e-3 MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1
  "TRAIN.swa_start_frac=[0.6, 0.7, 0.8]" DATASET.train_on_val=true
)
name=r50_all_e75m2_lr10
echo "[$(date '+%F %T')] START $name" >> "$LOG"
"$PY" train.py "${BASE[@]}" SEED=42 EXP_NAME="$name" > "logs/${name}_train.log" 2>&1
d=$(ls -dt "outputs/${name}"/*/ 2>/dev/null | head -1)
echo "[$(date '+%F %T')] DONE $name -> $d" >> "$LOG"
"$PY" scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] queue_local_lr 结束" >> "$LOG"
