#!/usr/bin/env bash
# 第三十七批（本地 3090）：TokenLearner replace 出口的正则消融 + 逐 token 辅助 CE。
#
# 为什么是这四个：远程 batch6/tl2 已覆盖「出口 × 正则强度」的主要格子，但缺三件事：
#   1) JS 项单独作用（不含熵约束）—— 决定「背景漂移」到底该归咎于哪一项；
#   2) 单侧熵下限单独作用（不含 JS）—— 同一问题的另一半；
#   3) 逐 token 辅助 CE（新增机制）：每个部件 token 都要过同一个分类头单独认对，
#      背景特征不可能单独分类，因此它把「分化到背景」这条路堵死，而把「是否分化」
#      仍留给 JS 项。若它与 JS 组合能把富集比拉回 ≥1.5× 且准确率不掉，这就是
#      TL 线的参数基准骨架。
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
  [ -n "$d" ] && "$PY" scripts/probe_mask_on_object.py --checkpoint "${d}swa_f0.5.pth" --n 96 \
      > "logs/${name}_obj.log" 2>&1
}

# 1) 只加逐 token 辅助 CE，不做任何分化约束
run r50_tl_rep_aux MODEL.tokenlearner=true MODEL.tl_replace=true MODEL.tl_dropout=0.0 \
    TRAIN.tl_aux_w=0.5
# 2) 单侧熵下限 + JS + 逐 token 辅助 CE（预期的基准骨架）
run r50_tl_rep_floor_aux MODEL.tokenlearner=true MODEL.tl_replace=true MODEL.tl_dropout=0.0 \
    MODEL.tl_ent_mode=floor MODEL.tl_ent_floor=0.5 MODEL.tl_js_w=0.5 MODEL.tl_js_target=0.35 \
    TRAIN.tl_reg_w=0.5 TRAIN.tl_aux_w=0.5
# 3) 只有 JS（不加熵、不加辅助 CE）：隔离出「JS 单独」的效应
run r50_tl_rep_js_only MODEL.tokenlearner=true MODEL.tl_replace=true MODEL.tl_dropout=0.0 \
    MODEL.tl_js_w=0.5 MODEL.tl_js_target=0.35 TRAIN.tl_reg_w=0.5
# 4) 只有单侧熵下限（不加 JS）：隔离出「floor 单独」的效应
run r50_tl_rep_floor_only MODEL.tokenlearner=true MODEL.tl_replace=true MODEL.tl_dropout=0.0 \
    MODEL.tl_ent_mode=floor MODEL.tl_ent_floor=0.5 TRAIN.tl_reg_w=0.5
echo "[$(date '+%F %T')] queueQQ 结束" >> "$LOG"
