#!/usr/bin/env bash
# 第三十八批（本地 3090）：训练期「特权裁剪」——用 CUB 关键点裁出的头部图当第二视图。
#
# 为什么插到最前（独立审查的结论）：在所有合法机制里，唯一有正证据的是**裁剪重编码**
# （cam_crop +0.79，是本项目唯一稳定正项）；而它改变的是「骨干看到的有效分辨率」，
# 不是 196 个格子上的权重分配。关键点裁出的头部比 CAM 框干净，是同一通道的更优输入。
# 训练时多一次前向是允许的；测试仍是单图单前向（模型 eval 不走 cam_aux 分支）。
#
# 四个格子（都在 baseline 配方上只改教师/标签/一致性）：
#   g0 gt_head + 硬标签（头部图是干净的原图裁剪，mixup 软目标是错误监督）
#   g1 gt_head + 软目标（对照组，量化「硬标签」这一项值多少）
#   g2 gt_head + 一致性 0.5（基线的 0.3 是给 CAM 框调的）
#   g3 gt_head + 一致性 0（只看裁剪分支自己的 CE，把「多一个视角」与「一致性」分开）
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
  [ -n "$d" ] && "$PY" scripts/probe_mask_on_object.py --checkpoint "${d}swa_f0.5.pth" --n 96 \
      > "logs/${name}_obj.log" 2>&1
}

run r50_gt_head_hl TRAIN.cam_teacher=gt_head TRAIN.cam_crop_hard_label=true
run r50_gt_head_soft TRAIN.cam_teacher=gt_head
run r50_gt_head_c05 TRAIN.cam_teacher=gt_head TRAIN.cam_crop_hard_label=true \
    TRAIN.cam_consistency=0.5
run r50_gt_head_nocons TRAIN.cam_teacher=gt_head TRAIN.cam_crop_hard_label=true \
    TRAIN.cam_consistency=0.0
echo "[$(date '+%F %T')] queueRR 结束" >> "$LOG"
