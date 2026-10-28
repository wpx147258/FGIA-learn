#!/usr/bin/env bash
# 第三十批：多区域重编码 + 学习式路由（训练），随后做 OOF 门控分析。
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
echo "[$(date '+%F %T')] START r50_region_route" >> "$LOG"
"$PY" train.py "${BASE_ARGS[@]}" \
  MODEL.region_route=true MODEL.rr_regions=2 TRAIN.rr_region_w=0.2 \
  EXP_NAME=r50_region_route > logs/r50_region_route_train.log 2>&1
d=$(ls -dt outputs/r50_region_route/*/ | head -1)
"$PY" scripts/test_tta.py --checkpoint "${d}swa_f0.5.pth" --views base \
  --dump dumps/single_r50_region_route >> "$LOG" 2>&1
echo "[$(date '+%F %T')] EVAL r50_region_route -> $d" >> "$LOG"
"$PY" scripts/fit_router_oof.py --checkpoint "${d}swa_f0.5.pth" \
  --dump dumps/rr_oof >> logs/rr_oof.log 2>&1
echo "[$(date '+%F %T')] OOF_DONE" >> "$LOG"
