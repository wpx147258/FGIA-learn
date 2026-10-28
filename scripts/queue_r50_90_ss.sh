#!/usr/bin/env bash
# 第三十九批（本地 3090）：GT 特权裁剪的两个修正。
#
# 诊断（本地 r50_gt_head_hl 的日志）：头框 4~6% 画面，裁剪分支的 CE 一直在 1.7~2.1，
# 而基线 CAM 框（约 10.6%）只要 1.2 —— 头裁得太紧，分类任务更难、监督更噪。
# 独立审查也给了同一个方向的数字：head-only 62.34、body 71.07（纯头缺颈/胸上下文）。
#
# 两格：
#   g0 part_pad=0.8：每边外扩 80%（面积约 2.6 倍，≈12~15%，接近 CAM 框），把颈/胸带进来
#   g1 同上 + 裁剪分支 CE 权重 0.4（基线是 0.2；GT 框比 CAM 框干净，可以多给一点）
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
  TRAIN.cam_teacher=gt_head TRAIN.cam_crop_hard_label=true
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

run r50_gth_pad08 TRAIN.part_pad=0.8
run r50_gth_pad08_w04 TRAIN.part_pad=0.8 TRAIN.cam_crop=0.4
echo "[$(date '+%F %T')] queueSS 结束" >> "$LOG"
