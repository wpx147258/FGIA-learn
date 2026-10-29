#!/usr/bin/env bash
# 全训基线的第二粒（SEED=1），跑在本机 3090。
#
# 为什么不是又一格 PA：现在所有 PA 结论都要跟全训基线比，而**当前代码分支上的全训基线
# 只有一粒 89.61**（另有四粒 89.94/89.92/89.70/89.64 是更早的代码）。而 PA 最好的
# all_pa4k8 = 89.75、AttnPool-CLS 两粒 89.85/89.52——都在基线种子散布（±0.2）之内，
# 说不清是不是增益。多一粒同代码的基线，比再多一格 PA 变体更能定这件事。
#
# 同时这也是本机的锚：attnpcls_s1(89.52) / e75m2_s7 都在本机跑，没有本机基线就没法解释。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/queue_r50_90.log
echo "[$(date '+%F %T')] queue_all_base_s1 等待 repro2 结束 ..." >> "$LOG"
until grep -qs "queue_pa4js_repro_test 结束" "$LOG"; do sleep 30; done

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
)
name=r50_all_base_s1
echo "[$(date '+%F %T')] START $name" >> "$LOG"
"$PY" train.py "${BASE[@]}" SEED=1 EXP_NAME="$name" > "logs/${name}_train.log" 2>&1
d=$(ls -dt "outputs/${name}"/*/ 2>/dev/null | head -1)
echo "[$(date '+%F %T')] DONE $name -> $d" >> "$LOG"
"$PY" scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] queue_all_base_s1 结束" >> "$LOG"
