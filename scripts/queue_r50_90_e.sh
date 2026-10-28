#!/usr/bin/env bash
# 统一队列：等 camfuse_m16 跑完 -> 评测 -> 竞争式部件划分（优先）-> 控制臂 -> 其余。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/queue_r50_90.log

BASE_ARGS=(
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

# 等 camfuse_m16 的 SWA 全部写完（swa_f0.7 是最后一个）
echo "[$(date '+%F %T')] queueE 等 camfuse_m16 ..." >> "$LOG"
until ls outputs/r50_camfuse_m16/*/swa_f0.7.pth >/dev/null 2>&1; do sleep 20; done
d=$(ls -dt outputs/r50_camfuse_m16/*/ | head -1)
"$PY" scripts/test_tta.py --checkpoint "${d}swa_f0.5.pth" --views base \
  --dump dumps/single_r50_camfuse_m16 >> "$LOG" 2>&1
echo "[$(date '+%F %T')] EVAL r50_camfuse_m16" >> "$LOG"

# 1) 竞争式部件划分（layer3 + 每格归属唯一部件 + 负载均衡 + 空间归一化）
run r50_pa2 MODEL.part_attn=true MODEL.pa_from=layer3 MODEL.pa_parts=4 \
    MODEL.pa_compete=true MODEL.pa_norm_spatial=true \
    TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0
# 2) 控制臂：同监督、同参数预算，去掉部件间自注意力
run r50_pa2_ctrl MODEL.part_attn=true MODEL.pa_from=layer3 MODEL.pa_parts=4 \
    MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_depth=0 \
    TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0
# 3) 三路融合
run r50_camfuse3 MODEL.cam_fuse=true MODEL.cam_fuse_regions=2 TRAIN.cam_fuse_w=1.0 TRAIN.cam_margin2=1.9
# 4) 骨干内注意力 / CLIP 池化对照
run r50_gc MODEL.global_ctx=true TRAIN.gc_lr=5.0e-4
run r50_clip_gap MODEL.name=resnet50_clip_gap.openai DATASET.crop_pct=0.875
run r50_clip_ap  MODEL.name=resnet50_clip.openai     DATASET.crop_pct=0.875

echo "[$(date '+%F %T')] queueE 结束" >> "$LOG"
