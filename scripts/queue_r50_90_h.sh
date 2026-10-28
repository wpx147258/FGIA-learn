#!/usr/bin/env bash
# 第七批：把「部件重新编码」（最有希望产生新信息的机制）提前，其余往后放。
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

# 通道硬分组（MA-CNN 的核心）：最后一种「同一张图上的推开」变体
run r50_pa4 MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=4 \
    MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 \
    TRAIN.cam_crop=0.2 TRAIN.cam_consistency=0.3 TRAIN.cam_margin=1.3 \
    TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0
# 部件裁出来重新编码 + 部件间自注意力（1+2 次前向），不带 CAM 分支以隔离变量
run r50_rerank2 MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=2 \
    MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 \
    MODEL.part_rerank=true MODEL.pr_margin=1.3 MODEL.pr_alpha=0.2 \
    TRAIN.cam_crop=0.0 TRAIN.cam_consistency=0.0 \
    TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0
# 控制臂：同样的两次裁剪与参数预算，但去掉部件间自注意力（逐点 MLP）
run r50_rerank2_ctrl MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=2 \
    MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 \
    MODEL.part_rerank=true MODEL.pr_margin=1.3 MODEL.pr_alpha=0.2 \
    TRAIN.cam_crop=0.0 TRAIN.cam_consistency=0.0 \
    TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0
# 三路融合
run r50_camfuse3 MODEL.cam_fuse=true MODEL.cam_fuse_regions=2 TRAIN.cam_fuse_w=1.0 TRAIN.cam_margin2=1.9
# 骨干内注意力 / CLIP 池化对照
run r50_gc MODEL.global_ctx=true TRAIN.gc_lr=5.0e-4
run r50_clip_gap MODEL.name=resnet50_clip_gap.openai DATASET.crop_pct=0.875
run r50_clip_ap  MODEL.name=resnet50_clip.openai     DATASET.crop_pct=0.875

echo "[$(date '+%F %T')] queueH 结束" >> "$LOG"
