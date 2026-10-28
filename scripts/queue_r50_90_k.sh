#!/usr/bin/env bash
# 第八批（优先级重排）：假设条件重检（top-1/top-2 两个 CAM 区域重编码 + 学习式融合）
# -> 通道分组 -> 部件重新编码。等 pa3 的 SWA 写完先补评测。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/queue_r50_90.log

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

echo "[$(date '+%F %T')] queueK 等 pa3 的 SWA ..." >> "$LOG"
until ls outputs/r50_pa3/*/swa_f0.7.pth >/dev/null 2>&1; do sleep 20; done
d=$(ls -dt outputs/r50_pa3/*/ | head -1)
"$PY" scripts/test_tta.py --checkpoint "${d}swa_f0.5.pth" --views base \
  --dump dumps/single_r50_pa3 >> "$LOG" 2>&1
echo "[$(date '+%F %T')] EVAL r50_pa3" >> "$LOG"

# 1) 假设条件重检：整图 + top1/top2 两个 CAM 区域（3 次前向），β 学习
run r50_hypfuse MODEL.hyp_fuse=true TRAIN.hyp_aux_w=0.2 \
    TRAIN.cam_crop=0.0 TRAIN.cam_consistency=0.0
# 2) 通道分组（同一张图上的最后一种结构性推开）
run r50_pa4 MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=4 \
    MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 \
    TRAIN.cam_crop=0.2 TRAIN.cam_consistency=0.3 TRAIN.cam_margin=1.3 \
    TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0
# 3) 部件重新编码（NTS 式框 + 注意力融合）
run r50_rerank2 MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=2 \
    MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 \
    MODEL.part_rerank=true MODEL.pr_margin=1.3 MODEL.pr_alpha=0.2 \
    TRAIN.cam_crop=0.0 TRAIN.cam_consistency=0.0 \
    TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0

echo "[$(date '+%F %T')] queueK 结束" >> "$LOG"
