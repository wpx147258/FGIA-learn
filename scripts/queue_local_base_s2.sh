#!/usr/bin/env bash
# 本机：全训基线 **SEED=2**。
#
# 为什么要这一格：`e75m2` 是目前唯一出现显著正增益的配置（本地 SEED=7：+0.62，p=0.02），
# 但远程 SEED=42 只有 +0.09。现在本地正在跑 e75m2_s1（对 base_s1=89.26 是第二对）。
# 这一格补第三粒种子，等 e75m2_s1 出分后如果方向为正，就再接 e75m2_s2 组成第三对；
# 若 e75m2_s1 掉回 89.2，这一格仍然是有用的基线种子（基线也需要 ≥3 粒才能定方差）。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/queue_r50_90.log
echo "[$(date '+%F %T')] queue_local_base_s2 等待 e75m2_s1 结束 ..." >> "$LOG"
until grep -qs "queue_local_e75m2_s1 结束" "$LOG"; do sleep 30; done
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
name=r50_all_base_s2
echo "[$(date '+%F %T')] START $name" >> "$LOG"
"$PY" train.py "${BASE[@]}" SEED=2 EXP_NAME="$name" > "logs/${name}_train.log" 2>&1
d=$(ls -dt "outputs/${name}"/*/ 2>/dev/null | head -1)
echo "[$(date '+%F %T')] DONE $name -> $d" >> "$LOG"
"$PY" scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] queue_local_base_s2 结束" >> "$LOG"
