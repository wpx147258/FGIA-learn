#!/usr/bin/env bash
# pa4_js（通道分组 K=4 + 掩码级 JS 正则）——val 最高的配置（89.91），但 test 只有 88.82 且是单 seed，
# 且原 checkpoint 已在磁盘清理中被删，无法复查 SWA 因子。这里重跑并同时评测 f0.5/0.6/0.7。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/queue_r50_90.log
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
  "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]"
  MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=4 MODEL.pa_grouped=true
  MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1
  MODEL.pa_js_w=0.5 MODEL.pa_js_target=0.35
  TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0
)
run () {
  local name="$1"; shift
  echo "[$(date '+%F %T')] START $name" >> "$LOG"
  "$PY" train.py "${BASE[@]}" "$@" EXP_NAME="$name" > "logs/${name}_train.log" 2>&1
  local d; d=$(ls -dt "outputs/${name}"/*/ 2>/dev/null | head -1)
  echo "[$(date '+%F %T')] DONE $name -> $d" >> "$LOG"
  for f in 0.5 0.6 0.7; do
    [ -f "${d}swa_f${f}.pth" ] && "$PY" scripts/test_tta.py --checkpoint "${d}swa_f${f}.pth" --views base \
        --dump "dumps/single_${name}_f${f}" >> "$LOG" 2>&1 && \
        echo "[$(date '+%F %T')] eval $name f$f done" >> "$LOG"
  done
}
run r50_pa4js_r2 SEED=7
echo "[$(date '+%F %T')] queue_pa4js 结束" >> "$LOG"
