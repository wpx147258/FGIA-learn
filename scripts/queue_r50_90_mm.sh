#!/usr/bin/env bash
# 第三十四批：两种注意力池化（TokenLearner / AttnPool）× 读出方式 × 深度 的对照矩阵。
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
}
echo "[$(date '+%F %T')] queueMM 等 queueLL ..." >> "$LOG"
until grep -qs "queueLL 结束" "$LOG"; do sleep 30; done
# TokenLearner：token 间自注意力（mean 读出）
run r50_tl_attn_d1 MODEL.tokenlearner=true MODEL.tl_attn_depth=1 MODEL.tl_readout=mean
# TokenLearner：cls 读出，depth 1 / 2
run r50_tl_cls_d1 MODEL.tokenlearner=true MODEL.tl_attn_depth=1 MODEL.tl_readout=cls
run r50_tl_cls_d2 MODEL.tokenlearner=true MODEL.tl_attn_depth=2 MODEL.tl_readout=cls
# AttnPool：cls 读出 depth 2（与 queueKK 里的 depth 1 配对）
run r50_attnp_cls_d2 MODEL.attn_pool=true MODEL.attn_readout=cls MODEL.attn_depth=2
echo "[$(date '+%F %T')] queueMM 结束" >> "$LOG"
