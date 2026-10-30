#!/usr/bin/env bash
# 2026-10-31：e75m2 的第七对配对（本机 SEED=5）。唯一站得住的增益就这一条，
# 多一对同机同种子配对能把五对合并的 p=0.0055 再压一档。
# 顺序：base_s5（50 轮，约 50 分钟）→ e75m2_s5（75 轮，约 75 分钟）→ 各自评测。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/queue_r50_90.log
BASE=(
  --config configs/baseline_config.yaml --set
  DATASET.image_size=448 TRAIN.batch_size=16
  TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05
  TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3
  TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0
  TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01
  TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05
  TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3
  MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1
  DATASET.train_on_val=true
)
run () { local name=$1; shift
  echo "[$(date '+%F %T')] START $name" >> "$LOG"
  "$PY" train.py "${BASE[@]}" "$@" EXP_NAME="$name" > "logs/${name}_train.log" 2>&1
  local d=$(ls -dt "outputs/${name}"/*/ 2>/dev/null | head -1)
  echo "[$(date '+%F %T')] DONE $name -> $d" >> "$LOG"
  [ -n "$d" ] && "$PY" scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1
}
run r50_all_base_s5  TRAIN.epochs=50 "TRAIN.swa_start_frac=[0.5,0.6,0.7]" SEED=5
run r50_all_e75m2_s5 TRAIN.epochs=75 "TRAIN.swa_start_frac=[0.6,0.7,0.8]" TRAIN.cam_margin2=1.6 TRAIN.cam_consistency2=0.15 SEED=5
echo "[$(date '+%F %T')] queue_local_s5_pair 结束" >> "$LOG"
