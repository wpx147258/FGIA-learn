#!/usr/bin/env bash
# 2026-10-30：`pa4k8_noreg`（PA 无任何正则）拿到 89.82，是全项目 PA 最高单粒、高于基线 +0.19。
# 但只有一粒、且配对 p=0.49，必须换种子复现才能下结论。
# 参照：同 SEED=7 的 pa4k8（有正则）= 89.45；同机基线 base_s7 = 89.07。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/queue_r50_90.log
echo "[$(date '+%F %T')] queue_local_noreg_s7 等待 cmp05_s1 结束 ..." >> "$LOG"
while pgrep -f "EXP_NAME=r50_all_pa4k8_cmp05_s1$" >/dev/null 2>&1; do sleep 30; done
d=$(ls -dt outputs/r50_all_pa4k8_cmp05_s1/*/ 2>/dev/null | head -1)
[ -n "$d" ] && { echo "[$(date '+%F %T')] eval r50_all_pa4k8_cmp05_s1" >> "$LOG"; "$PY" scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1; }

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
  TRAIN.pa_div_w=0.0 TRAIN.pa_aux_w=0.0 MODEL.pa_js_w=0.0 TRAIN.pa_lr=5.0e-4
)
name=r50_all_pa4k8_noreg_s7
echo "[$(date '+%F %T')] START $name" >> "$LOG"
"$PY" train.py "${BASE[@]}" SEED=7 EXP_NAME="$name" > "logs/${name}_train.log" 2>&1
d=$(ls -dt "outputs/${name}"/*/ 2>/dev/null | head -1)
echo "[$(date '+%F %T')] DONE $name -> $d" >> "$LOG"
"$PY" scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] queue_local_noreg_s7 结束" >> "$LOG"
