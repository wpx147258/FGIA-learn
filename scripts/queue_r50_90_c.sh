#!/usr/bin/env bash
# 第三批（重排）：stride8 已被证伪（88.56 vs 89.40），不再往后排队。
# 优先跑「多区域再编码 + 学习式融合」（文献里真正有效的机制），其次是骨干内注意力与 CLIP 池化对照。
# 用日志里的「训练结束」判断前一个实验是否跑完，不用 pgrep（避免匹配到自己的监控命令）。
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

eval_run () {   # $1 = 实验名
  local d
  d=$(ls -dt "outputs/$1"/*/ 2>/dev/null | head -1)
  [ -n "$d" ] || return 0
  "$PY" scripts/test_tta.py --checkpoint "${d}swa_f0.5.pth" --views base \
    --dump "dumps/single_$1" >> "$LOG" 2>&1
  echo "[$(date '+%F %T')] EVAL $1 -> $d" >> "$LOG"
}

run () {
  local name="$1"; shift
  echo "[$(date '+%F %T')] START $name" >> "$LOG"
  "$PY" train.py "${BASE_ARGS[@]}" "$@" EXP_NAME="$name" > "logs/${name}_train.log" 2>&1
  echo "[$(date '+%F %T')] DONE $name" >> "$LOG"
  eval_run "$name"
}

# 1) 等正在跑的 r50_pa_aux 结束（它是上一个队列的子进程，父进程已被我停掉）
echo "[$(date '+%F %T')] queue3 等 r50_pa_aux ..." >> "$LOG"
until grep -qs "训练结束" outputs/r50_pa_aux/*/log.txt; do sleep 20; done
eval_run r50_pa_aux

# 2) 双路融合：整图 + CAM 框放大（测试 2 次前向，β 训练得到）
run r50_camfuse MODEL.cam_fuse=true TRAIN.cam_fuse_w=1.0
# 3) 同上，框外扩 1.6（更宽的上下文）
run r50_camfuse_m16 MODEL.cam_fuse=true TRAIN.cam_fuse_w=1.0 TRAIN.cam_margin=1.6
# 4) layer3 -> layer4 之间的全局自注意力
run r50_gc MODEL.global_ctx=true TRAIN.gc_lr=5.0e-4
# 5) CLIP RN50 同权重、只差池化头（机制对照）
run r50_clip_gap MODEL.name=resnet50_clip_gap.openai DATASET.crop_pct=0.875
run r50_clip_ap  MODEL.name=resnet50_clip.openai     DATASET.crop_pct=0.875

echo "[$(date '+%F %T')] queue3 结束" >> "$LOG"
