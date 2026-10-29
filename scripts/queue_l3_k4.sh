#!/usr/bin/env bash
# 重跑 pa4_l3_k4（layer3 出掩码，K=4）。09:32 那一次在 ep12 被抢占脚本误杀：
# 抢占的 grep 匹配到了 09:21 那次**失败的启动**留下的 "START r50_pa4_l3_k8" 旧行。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/queue_r50_90.log
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
  "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]"
  MODEL.part_attn=true MODEL.pa_from=layer3 MODEL.pa_grouped=true
  MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1
  MODEL.pa_depth=1 TRAIN.pa_div_w=1.0 TRAIN.pa_aux_w=0.25 TRAIN.pa_lr=5.0e-4
  MODEL.pa_parts=4
)
# 先补一格「更极端的集中」：熵上限 0.5（≈1.6 格），用户判断权重应更集中
# 对照：同样的紧上限、**不退火**——用来量退火到底值不值
run r50_tl_ceil05_noramp MODEL.tokenlearner=true MODEL.tl_dropout=0.0 MODEL.tl_replace=true \
    MODEL.tl_ent_mode=ceiling MODEL.tl_ent_ceiling=0.5 MODEL.tl_js_w=0.5 MODEL.tl_js_target=0.05 \
    TRAIN.tl_reg_w=0.5 TRAIN.tl_aux_w=0.5
# 更紧一档：ceil 0.3（≈1.35 格）
run r50_tl_ceil03_noramp MODEL.tokenlearner=true MODEL.tl_dropout=0.0 MODEL.tl_replace=true \
    MODEL.tl_ent_mode=ceiling MODEL.tl_ent_ceiling=0.3 MODEL.tl_js_w=0.5 MODEL.tl_js_target=0.05 \
    TRAIN.tl_reg_w=0.5 TRAIN.tl_aux_w=0.5

name=r50_pa4_l3_k4b
echo "[$(date '+%F %T')] START $name" >> "$LOG"
"$PY" train.py "${BASE[@]}" EXP_NAME="$name" > "logs/${name}_train.log" 2>&1
d=$(ls -dt "outputs/${name}"/*/ 2>/dev/null | head -1)
echo "[$(date '+%F %T')] DONE $name -> $d" >> "$LOG"
[ -n "$d" ] && "$PY" scripts/test_tta.py --checkpoint "${d}swa_f0.5.pth" --views base \
    --dump "dumps/single_${name}" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] queue_l3_k4 结束" >> "$LOG"
