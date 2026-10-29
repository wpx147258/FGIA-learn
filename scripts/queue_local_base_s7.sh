#!/usr/bin/env bash
# 本机：全训基线 **SEED=7**。
#
# 为什么这一格最划算：本机已经/将要跑完两格 SEED=7 的实验——
#   `r50_all_e75m2_s7`（75ep + 双尺度，f0.6 = 89.68）
#   `r50_all_attnpcls_s7`（AttnPool-CLS）
# 但**没有 SEED=7 的 50-epoch 基线**，所以这两格现在都无法解释。
# 补这一格之后：
#   base_s7 vs attnpcls_s7  → 同机同种子的 AttnPool 单变量对照
#   base_s7 vs e75m2_s7     → 同机同种子的"长训练+双尺度"单变量对照
# 一格激活两对比较，比再跑一个无关的变体更值。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/queue_r50_90.log
echo "[$(date '+%F %T')] queue_local_base_s7 等待 attnpcls_s7 结束 ..." >> "$LOG"
until grep -qs "queue_local_attnpcls_s7 结束" "$LOG"; do sleep 30; done
BASE=(
  --config configs/baseline_config.yaml --set
  DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16
  TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05
  TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3
  TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0
  TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01
  TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05
  TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3
  MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1
  "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]" DATASET.train_on_val=true
)
name=r50_all_base_s7
echo "[$(date '+%F %T')] START $name" >> "$LOG"
"$PY" train.py "${BASE[@]}" SEED=7 EXP_NAME="$name" > "logs/${name}_train.log" 2>&1
d=$(ls -dt "outputs/${name}"/*/ 2>/dev/null | head -1)
echo "[$(date '+%F %T')] DONE $name -> $d" >> "$LOG"
"$PY" scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] queue_local_base_s7 结束" >> "$LOG"
