#!/usr/bin/env bash
# 全训 + PartAttention K=2（compete=false）：K 从未扫过 2。
# 动机：K=4/8 的部件是"随机 1/K 划分"；K=2 是唯一能退化成"前景/背景两块重叠掩码"的设置，
#       与"部件天然重叠、不该互斥"的判断直接对应，且成本一样。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/queue_r50_90.log
echo "[$(date '+%F %T')] queue_all_k2 等待 repro2 结束 ..." >> "$LOG"
until grep -qs "queue_pa4js_repro_test 结束" "$LOG"; do sleep 30; done

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
  MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=2 MODEL.pa_grouped=true
  MODEL.pa_compete=false MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1
  TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0
)
name=r50_all_k2
echo "[$(date '+%F %T')] START $name" >> "$LOG"
"$PY" train.py "${BASE[@]}" SEED=42 EXP_NAME="$name" > "logs/${name}_train.log" 2>&1
d=$(ls -dt "outputs/${name}"/*/ 2>/dev/null | head -1)
echo "[$(date '+%F %T')] DONE $name -> $d" >> "$LOG"
"$PY" scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] queue_all_k2 结束" >> "$LOG"
