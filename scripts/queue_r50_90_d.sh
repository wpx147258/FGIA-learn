#!/usr/bin/env bash
# 第四批（改）：先把「竞争式部件划分（把部件推开）」跑掉，再跑三路融合。
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

echo "[$(date '+%F %T')] queue4 等第三批 ..." >> "$LOG"
until grep -qs "queue3 结束" "$LOG"; do sleep 30; done

# 1) 竞争式部件划分：每个格子只属于一个部件 + 负载均衡 + 空间归一化，检测器在 layer3(28×28)
run r50_pa2 MODEL.part_attn=true MODEL.pa_from=layer3 MODEL.pa_parts=4 \
    MODEL.pa_compete=true MODEL.pa_norm_spatial=true \
    TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0
# 2) 控制臂：监督与参数预算相同，但去掉部件间自注意力（逐点 MLP）
run r50_pa2_ctrl MODEL.part_attn=true MODEL.pa_from=layer3 MODEL.pa_parts=4 \
    MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_depth=0 \
    TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0
# 3) 三路融合：整图 + 紧裁剪 + 宽裁剪
run r50_camfuse3 MODEL.cam_fuse=true MODEL.cam_fuse_regions=2 TRAIN.cam_fuse_w=1.0 TRAIN.cam_margin2=1.9

echo "[$(date '+%F %T')] queue4 结束" >> "$LOG"
