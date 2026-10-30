#!/usr/bin/env bash
# 2026-10-30：用户判断——真实部位天然重叠（眼睛在头部里、翅根连身体），
# 所以**不该用 compete 做硬划分**（它强制每位置 Σ_k m ≡ 1，物理上禁止重叠）。
# 新配置：sigmoid 掩码（允许重叠）+ 空间紧致（每个部件成片）+ 反重复铰链（只罚几乎相同的部件对）。
#
# 本地 SEED=1 上形成三格干净的消融链（同机同种子）：
#   pa4k8_s1      compete=true  无紧致            → 89.35（已跑）
#   cmp05_s1      compete=true  +紧致 0.5         → 在跑
#   ovl05_s1      compete=false +紧致 0.5 +反重复 → 本脚本
# 基线 base_s1 = 89.26。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/queue_r50_90.log
echo "[$(date '+%F %T')] queue_local_overlap 等待 cmp05_s1 结束 ..." >> "$LOG"
while pgrep -f "EXP_NAME=r50_all_pa4k8_cmp05_s1$" >/dev/null 2>&1; do sleep 30; done
# 先把正在跑的那格评掉（它是直接 train.py 起的，没有自动评测）
d=$(ls -dt outputs/r50_all_pa4k8_cmp05_s1/*/ 2>/dev/null | head -1)
[ -n "$d" ] && { echo "[$(date '+%F %T')] eval r50_all_pa4k8_cmp05_s1" >> "$LOG"; "$PY" scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1; }

BASE=(
  --config configs/baseline_config.yaml --set
  DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16
  TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05
  TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3
  TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0
  TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01
  TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05
  TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3
  MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1
  "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]" DATASET.train_on_val=true
  MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=8 MODEL.pa_grouped=true
  MODEL.pa_compete=false MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1
  MODEL.pa_compact_w=0.5 MODEL.pa_dup_w=0.5 MODEL.pa_dup_thr=0.8
  TRAIN.pa_div_w=0.0 TRAIN.pa_aux_w=0.5 TRAIN.pa_lr=5.0e-4
)
name=r50_all_pa4k8_ovl05_s1
echo "[$(date '+%F %T')] START $name" >> "$LOG"
"$PY" train.py "${BASE[@]}" SEED=1 EXP_NAME="$name" > "logs/${name}_train.log" 2>&1
d=$(ls -dt "outputs/${name}"/*/ 2>/dev/null | head -1)
echo "[$(date '+%F %T')] DONE $name -> $d" >> "$LOG"
"$PY" scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] queue_local_overlap 结束" >> "$LOG"
