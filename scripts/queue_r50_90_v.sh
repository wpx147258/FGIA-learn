#!/usr/bin/env bash
# 第十九批：按调优顾问的 P0/P1 顺序跑 pa4 的止损与改出口实验。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/queue_r50_90.log

PA_BASE=(
  MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=4
  MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1
  TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0
  TRAIN.cam_crop=0.2 TRAIN.cam_consistency=0.3 TRAIN.cam_margin=1.3
)
BASE_ARGS=(
  --config configs/baseline_config.yaml --set
  DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16
  TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05
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

echo "[$(date '+%F %T')] queueV 等 r50_long75 ..." >> "$LOG"
until ls outputs/r50_long75/*/swa_f0.7.pth >/dev/null 2>&1; do sleep 30; done
d=$(ls -dt outputs/r50_long75/*/ | head -1)
"$PY" scripts/test_tta.py --checkpoint "${d}swa_f0.5.pth" --views base \
  --dump dumps/single_r50_long75 >> "$LOG" 2>&1
echo "[$(date '+%F %T')] EVAL r50_long75" >> "$LOG"

# P0-1 控制臂：去掉部件间自注意力（参数对齐）
run r50_pa4_ctrl "${PA_BASE[@]}" MODEL.pa_depth=0
# P0-2 残差幅度控制：固定为 GAP 范数的 0.3 倍 + 打开 weight decay
run r50_pa4_res "${PA_BASE[@]}" MODEL.pa_res_scale=0.3 TRAIN.pa_wd=0.05
# P1-1 改出口：去掉残差，部件证据走 logit 级融合（权重绑定主头）
run r50_pa4_logit "${PA_BASE[@]}" MODEL.pa_logit_fuse=true MODEL.pa_no_residual=true \
    TRAIN.pa_wd=0.05
# P1-2 K=8（每组 256 通道）
run r50_pa4_k8 MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=8 \
    MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 \
    TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.25 TRAIN.pa_div_w=1.0 \
    TRAIN.cam_crop=0.2 TRAIN.cam_consistency=0.3 TRAIN.cam_margin=1.3

echo "[$(date '+%F %T')] queueV 结束" >> "$LOG"
