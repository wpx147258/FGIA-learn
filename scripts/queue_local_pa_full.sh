#!/usr/bin/env bash
# 2026-10-30：用户要求排「四项完整」的 PA —— 这是唯一有资格回答"成形后的部件有没有用"的配置。
#
#   pa_compact_w = 3.0   空间紧致（框 51%→目标 <25%）
#   pa_border_w  = 1.0   外圈质量惩罚（把团块从图像边缘赶回物体上）
#   pa_dup_w     = 1.0   反重复，判据改为**软 IoU**、阈值 0.9（允许眼睛/头部这种面积重叠，只禁近乎相同）
#   pa_compete   = false 允许重叠（用户判断：硬划分 Σ_k m≡1 物理上禁止重叠）
#   pa_div_w = 0, pa_aux_w = 0 —— 依据 13:05 的消融：这两个正则被证明无效（裸 PA 89.82 最高）
#
# 参照（本地同机同种子 SEED=1）：base 89.26、pa4k8_s1 89.35、cmp05_s1(紧致无边界) 89.14。
# 若这格低于 89.26 → 说明"成形"本身不带来增益；若高于 89.35 → 第一个"机制真的有用"的证据。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/queue_r50_90.log
echo "[$(date '+%F %T')] queue_local_pa_full 等待 noreg_s7 结束 ..." >> "$LOG"
while pgrep -f "EXP_NAME=r50_all_pa4k8_noreg_s7$" >/dev/null 2>&1; do sleep 30; done

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
  MODEL.pa_compete=false MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1
  MODEL.pa_compact_w=3.0 MODEL.pa_dup_w=1.0 MODEL.pa_dup_thr=0.9 MODEL.pa_border_w=1.0
  TRAIN.pa_div_w=0.0 TRAIN.pa_aux_w=0.0 MODEL.pa_js_w=0.0 TRAIN.pa_lr=5.0e-4
)
name=r50_all_pa4k8_full_s1
echo "[$(date '+%F %T')] START $name" >> "$LOG"
"$PY" train.py "${BASE[@]}" SEED=1 EXP_NAME="$name" > "logs/${name}_train.log" 2>&1
d=$(ls -dt "outputs/${name}"/*/ 2>/dev/null | head -1)
echo "[$(date '+%F %T')] DONE $name -> $d" >> "$LOG"
"$PY" scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] queue_local_pa_full 结束" >> "$LOG"
