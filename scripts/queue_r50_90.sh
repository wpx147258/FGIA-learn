#!/usr/bin/env bash
# R50 破 90 的队列（v3）：把「部件提取 + 部件间自注意力」（用户点名的 [70]–[76] 思路）
# 提到最前面，其余按审查意见修订后的顺序。
# 第 1 个 r50_s8 是手动起的；本脚本等它结束后依次跑后面的。
# 每个实验都做单视图评测并 dump 概率，供 scripts/compare_runs.py 做配对检验。
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
    # 单视图、单权重：--views base，只给一个 checkpoint（默认 --views 是四视图）
    "$PY" scripts/test_tta.py --checkpoint "${d}swa_f0.5.pth" --views base \
      --dump "dumps/single_${name}" >> "$LOG" 2>&1
  fi
}

echo "[$(date '+%F %T')] 等待 r50_s8 ..." >> "$LOG"
until ! pgrep -f "EXP_NAME=r50_s8" >/dev/null 2>&1; do sleep 20; done
d8=$(ls -dt outputs/r50_s8/*/ | head -1)
"$PY" scripts/test_tta.py --checkpoint "${d8}swa_f0.5.pth" --views base \
  --dump dumps/single_r50_s8 >> "$LOG" 2>&1
echo "[$(date '+%F %T')] r50_s8 评测完成 $d8" >> "$LOG"

# 1) 部件提取 + 部件间自注意力 + 部件级辅助监督（MA-CNN 的部件损失与散度项）
run r50_pa_aux  MODEL.part_attn=true MODEL.pa_parts=4 TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=0.1
# 2) 同结构但去掉两项辅助监督（消融：没有部件级监督时注意力会不会退化）
run r50_pa      MODEL.part_attn=true MODEL.pa_parts=4 TRAIN.pa_lr=5.0e-4
# 3) 控制臂：参数对齐（1.614M vs 1.615M）但**不做部件间混合**（逐点 MLP），监督相同
run r50_pa_ctrl MODEL.part_attn=true MODEL.pa_parts=4 MODEL.pa_depth=0 TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=0.1
# 4) CLIP RN50 + GAP   —— 与 5) 权重完全相同，只差池化头
run r50_clip_gap MODEL.name=resnet50_clip_gap.openai DATASET.crop_pct=0.875
# 5) CLIP RN50 + 预训练过的注意力池化（AttentionPool2d）
run r50_clip_ap  MODEL.name=resnet50_clip.openai     DATASET.crop_pct=0.875
# 6) layer3 -> layer4 之间的全局自注意力（不依赖 stride8）
run r50_gc      MODEL.global_ctx=true TRAIN.gc_lr=5.0e-4
# 7) GlobalContext 控制臂：同样参数量的逐点 MLP，没有 token 混合
run r50_gc_ctrl MODEL.global_ctx=true MODEL.gc_depth=0 TRAIN.gc_lr=5.0e-4
# 8) 只有 stride8 看起来有希望才值得跑（内容池化先不排，省时间）
run r50_s8_gc   MODEL.stride8=true MODEL.global_ctx=true TRAIN.gc_lr=5.0e-4

echo "[$(date '+%F %T')] 队列结束" >> "$LOG"
