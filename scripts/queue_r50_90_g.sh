#!/usr/bin/env bash
# 第六批：部件裁出来重新编码 + 部件间自注意力（MA-CNN 流水线，融合器换成注意力）。
# 测试 1+K 次前向（机制需要，不是 TTA）。等第五批结束后跑。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/queue_r50_90.log

BASE_ARGS=(
  --config configs/baseline_config.yaml --set
  DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16
  TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05
  TRAIN.cam_crop=0.0 TRAIN.cam_consistency=0.0
  TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01
  TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05
  TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3
  MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1
  "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]"
)
run () {
  local name="$1"; shift
  echo "[$(date '+%F %T')] START $name" >> "$LOG"
  "$PY" train.py "${BASE_ARGS[@]}" "$@" EXP_NAME="$name" > "logs/${name}_train.log" 2>&1
  local d; d=$(ls -dt "outputs/${name}"/*/ 2>/dev/null | head -1)
  echo "[$(date '+%F %T')] DONE $name -> $d" >> "$LOG"
  [ -n "$d" ] && "$PY" scripts/test_tta.py --checkpoint "${d}swa_f0.5.pth" --views base \
      --dump "dumps/single_${name}" >> "$LOG" 2>&1
}

echo "[$(date '+%F %T')] queueG 等第五批 ..." >> "$LOG"
until grep -qs "queueF 结束" "$LOG"; do sleep 30; done

# 通道硬分组（MA-CNN 的核心）：每个部件只看自己那组通道 + 硬分配
run r50_pa4 MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=4 \
    MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 \
    TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0
# 部件重新编码：K=2 个部件 -> 测试 3 次前向；检测器在 layer4，硬分配（tau=0.1）
run r50_rerank2 MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=2 \
    MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 \
    MODEL.part_rerank=true MODEL.pr_margin=1.3 MODEL.pr_alpha=0.2 \
    TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0

echo "[$(date '+%F %T')] queueG 结束" >> "$LOG"
