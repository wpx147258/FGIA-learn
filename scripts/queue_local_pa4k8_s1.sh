#!/usr/bin/env bash
# 本机 3090：全训 + PartAttention K=8（compete、无 JS），**SEED=1**。
#
# 为什么要这一格：远程的 all_pa4k8(89.79) 与 all_base(89.63) 是**不同种子**（都是 42，
# 但那是同一粒种子的两次运行），而且远程与本地数值本来就不同。本机已经有/将有
# `r50_all_base_s1`（SEED=1），这里再跑同一台机器、同一个 SEED=1 的 pa4k8 ——
# 两者只差一个部件注意力模块，是**同机同种子的单变量对照**，比跨机跨种子的比较干净。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/queue_r50_90.log
echo "[$(date '+%F %T')] queue_local_pa4k8_s1 等待 base_s1 结束 ..." >> "$LOG"
until grep -qs "queue_all_base_s1 结束" "$LOG"; do sleep 30; done

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
  MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=8 MODEL.pa_grouped=true
  MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1
  TRAIN.pa_div_w=1.0 TRAIN.pa_aux_w=0.5 TRAIN.pa_lr=5.0e-4
)
name=r50_all_pa4k8_s1
echo "[$(date '+%F %T')] START $name" >> "$LOG"
"$PY" train.py "${BASE[@]}" SEED=1 EXP_NAME="$name" > "logs/${name}_train.log" 2>&1
d=$(ls -dt "outputs/${name}"/*/ 2>/dev/null | head -1)
echo "[$(date '+%F %T')] DONE $name -> $d" >> "$LOG"
"$PY" scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] queue_local_pa4k8_s1 结束" >> "$LOG"
