#!/usr/bin/env bash
# 第二批：用户确认「机制需要的多次前向可以，单纯 TTA 不行」之后加的实验。
# 等第一批队列（scripts/queue_r50_90.sh）结束后再跑，避免两个队列抢显卡。
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
  local d
  d=$(ls -dt "outputs/${name}"/*/ 2>/dev/null | head -1)
  echo "[$(date '+%F %T')] DONE $name -> $d" >> "$LOG"
  if [ -n "$d" ]; then
    "$PY" scripts/test_tta.py --checkpoint "${d}swa_f0.5.pth" --views base \
      --dump "dumps/single_${name}" >> "$LOG" 2>&1
  fi
}

echo "[$(date '+%F %T')] queue2 等待第一批结束 ..." >> "$LOG"
until ! pgrep -f "bash scripts/queue_r50_90.sh" >/dev/null 2>&1; do sleep 30; done

# 双路融合：整图 + CAM 框放大，测试两次前向，融合系数 β 训练得到
run r50_camfuse MODEL.cam_fuse=true TRAIN.cam_fuse_w=1.0
# 同上，但裁剪框外扩更大（1.6），看融合的是"局部证据"还是"上下文"
run r50_camfuse_m16 MODEL.cam_fuse=true TRAIN.cam_fuse_w=1.0 TRAIN.cam_margin=1.6

echo "[$(date '+%F %T')] queue2 结束" >> "$LOG"
