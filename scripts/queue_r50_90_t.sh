#!/usr/bin/env bash
# 第十七批：在已证明有效的 CAM 机制上加「双尺度一致性 + 更长训练」，同时保留部件线的对照。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/queue_r50_90.log

BASE_ARGS=(
  --config configs/baseline_config.yaml --set
  DATASET.image_size=448 TRAIN.batch_size=16
  TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05
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

echo "[$(date '+%F %T')] queueT 等 r50_camaux2 ..." >> "$LOG"
until ls outputs/r50_camaux2/*/swa_f0.7.pth >/dev/null 2>&1; do sleep 30; done
d=$(ls -dt outputs/r50_camaux2/*/ | head -1)
"$PY" scripts/test_tta.py --checkpoint "${d}swa_f0.5.pth" --views base \
  --dump dumps/single_r50_camaux2 >> "$LOG" 2>&1
echo "[$(date '+%F %T')] EVAL r50_camaux2" >> "$LOG"

# 1) 双尺度 CAM 一致性 + 75 轮（在已验证的 CAM 机制上加，不加新模块）
run r50_long75 TRAIN.epochs=75 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 \
    TRAIN.cam_margin=1.3 TRAIN.cam_margin2=1.6 \
    TRAIN.cam_consistency=0.3 TRAIN.cam_consistency2=0.15 TRAIN.cam_consistency_T=2.0 \
    "TRAIN.swa_start_frac=[0.6, 0.7, 0.8]"
# 2) 训练期部件裁剪（掩码版，对照）
run r50_partaux MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=4 \
    MODEL.pa_grouped=true MODEL.pa_compete=false MODEL.pa_norm_spatial=true \
    MODEL.part_train_aux=true MODEL.pta_parts=2 \
    TRAIN.epochs=50 TRAIN.pta_aux_w=0.2 TRAIN.pta_cons_w=0.3 \
    TRAIN.cam_crop=0.2 TRAIN.cam_consistency=0.3 TRAIN.cam_margin=1.3 \
    TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=2.0
# 3) pa4 种子复现
run r50_pa4_s1 MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=4 \
    MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 \
    TRAIN.epochs=50 TRAIN.cam_crop=0.2 TRAIN.cam_consistency=0.3 TRAIN.cam_margin=1.3 SEED=1 \
    TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0
# 4) 学习式假设组合器
run r50_hypfuse2 MODEL.hyp_fuse=true MODEL.hyp_combiner=true TRAIN.hyp_aux_w=0.2 \
    TRAIN.epochs=50 TRAIN.cam_crop=0.0 TRAIN.cam_consistency=0.0
# 5) 其余
run r50_camfuse3 MODEL.cam_fuse=true MODEL.cam_fuse_regions=2 TRAIN.cam_fuse_w=1.0 \
    TRAIN.epochs=50 TRAIN.cam_margin=1.3 TRAIN.cam_margin2=1.9 \
    TRAIN.cam_crop=0.2 TRAIN.cam_consistency=0.3
run r50_gc MODEL.global_ctx=true TRAIN.gc_lr=5.0e-4 TRAIN.epochs=50
run r50_clip_gap MODEL.name=resnet50_clip_gap.openai DATASET.crop_pct=0.875 TRAIN.epochs=50
run r50_clip_ap  MODEL.name=resnet50_clip.openai     DATASET.crop_pct=0.875 TRAIN.epochs=50

echo "[$(date '+%F %T')] queueT 结束" >> "$LOG"
