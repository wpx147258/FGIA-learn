#!/usr/bin/env bash
# 2026-10-31：Random Erasing 的第二个种子（本地 SEED=1 / SEED=7 各一格，顺序跑）。
# 克隆上 `e75m2_rare` = 89.75（+0.07 vs e75m2_s42r 89.68），是全项目最高单粒，
# 但单粒在噪声内。本机已有 e75m2_s1=89.58 / e75m2_s7=89.68 作为同机同种子对照。
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
  TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01
  TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05
  TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3
  MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1
  "TRAIN.swa_start_frac=[0.6, 0.7, 0.8]" DATASET.train_on_val=true
  TRAIN.random_erasing=0.25
)
for seed in 1 7; do
  name=r50_all_e75m2_rare_s${seed}
  echo "[$(date '+%F %T')] START $name" >> "$LOG"
  "$PY" train.py "${BASE[@]}" SEED=$seed EXP_NAME="$name" > "logs/${name}_train.log" 2>&1
  d=$(ls -dt "outputs/${name}"/*/ 2>/dev/null | head -1)
  echo "[$(date '+%F %T')] DONE $name -> $d" >> "$LOG"
  [ -n "$d" ] && "$PY" scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1
done
echo "[$(date '+%F %T')] queue_local_rare 结束" >> "$LOG"
