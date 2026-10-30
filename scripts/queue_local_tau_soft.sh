#!/usr/bin/env bash
# 2026-10-30：τ 扫描显示**软化分区单调更好**（K=4：τ=0.05→89.30、0.10→89.54、0.20→89.73）。
# τ 增大就是允许部件部分重叠——正是"眼睛和头部天然重叠"的机制实现；
# 而 compete=false（彻底取消分区）会塌成同一张掩码（89.25）。
# 所以正确的方向是「软化」，不是「取消」。
#
# 这一批在 K=8（最好的 K）上继续软化：τ = 0.3 与 0.5。
# 参照（本地同机同 SEED=1）：pa4k8_s1（K=8, τ=0.1）= 89.35。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/queue_r50_90.log
echo "[$(date '+%F %T')] queue_local_tau_soft 等待 c30_s1 结束 ..." >> "$LOG"
while pgrep -f "EXP_NAME=r50_all_e75m2_c30_s1$" >/dev/null 2>&1; do sleep 30; done
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
  MODEL.pa_compete=true MODEL.pa_norm_spatial=true
  TRAIN.pa_div_w=1.0 TRAIN.pa_aux_w=0.5 TRAIN.pa_lr=5.0e-4
)
for tau in 0.3 0.5; do
  name=r50_all_pa4k8_tau${tau/./}_s1
  echo "[$(date '+%F %T')] START $name" >> "$LOG"
  "$PY" train.py "${BASE[@]}" MODEL.pa_tau=$tau SEED=1 EXP_NAME="$name" > "logs/${name}_train.log" 2>&1
  d=$(ls -dt "outputs/${name}"/*/ 2>/dev/null | head -1)
  echo "[$(date '+%F %T')] DONE $name -> $d" >> "$LOG"
  "$PY" scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1
done
echo "[$(date '+%F %T')] queue_local_tau_soft 结束" >> "$LOG"
