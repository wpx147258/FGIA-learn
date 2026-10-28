#!/usr/bin/env bash
# 远程第三批（2026-10-29 改版，插在 tl2 之后）：TokenLearner 逐 token 辅助 CE 的组合。
#
# 背景：§13 的两条退化都源于「多样性正则没有落点约束」。新增的 TRAIN.tl_aux_w
# 让每个 token 单独过同一个分类头算 CE —— 背景特征不可能单独分类，于是 JS 再怎么
# 把图推开，也推不到背景上。本轮的四个格子就是「锚 + 推开」的不同配比：
#   g0 只有锚+JS（无 floor）—— 检验 JS 会不会把图压成单点、需要地板吗
#   g1 锚加强（aux 1.0）
#   g2 地板抬高（floor 1.5，部件必须是区域）
#   g3 四个 token（S=4，维度压力更小）
set -u
cd /root/autodl-tmp/cub-bird || exit 1
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
mkdir -p logs dumps
BASE=(--config configs/baseline_config.yaml --set
  DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16
  TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05
  TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3
  TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0
  TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01
  TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05
  TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3
  MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1
  "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]")
launch () { local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" train.py "${BASE[@]}" "$@" EXP_NAME="$name" \
    > "logs/${name}_train.log" 2>&1 &
  echo "launched $name on GPU$gpu pid $!"
}
launch 0 r50_tl_aux_js MODEL.tokenlearner=true MODEL.tl_replace=true MODEL.tl_dropout=0.0 \
         MODEL.tl_js_w=0.5 MODEL.tl_js_target=0.35 TRAIN.tl_reg_w=0.5 TRAIN.tl_aux_w=0.5
launch 1 r50_tl_floor_aux_w1 MODEL.tokenlearner=true MODEL.tl_replace=true MODEL.tl_dropout=0.0 \
         MODEL.tl_ent_mode=floor MODEL.tl_ent_floor=0.5 MODEL.tl_js_w=0.5 MODEL.tl_js_target=0.35 \
         TRAIN.tl_reg_w=0.5 TRAIN.tl_aux_w=1.0
launch 2 r50_tl_floor15_aux MODEL.tokenlearner=true MODEL.tl_replace=true MODEL.tl_dropout=0.0 \
         MODEL.tl_ent_mode=floor MODEL.tl_ent_floor=1.5 MODEL.tl_js_w=0.5 MODEL.tl_js_target=0.35 \
         TRAIN.tl_reg_w=0.5 TRAIN.tl_aux_w=0.5
launch 3 r50_tl_floor_aux_s4 MODEL.tokenlearner=true MODEL.tl_replace=true MODEL.tl_dropout=0.0 \
         MODEL.tl_tokens=4 MODEL.tl_ent_mode=floor MODEL.tl_ent_floor=0.5 \
         MODEL.tl_js_w=0.5 MODEL.tl_js_target=0.35 TRAIN.tl_reg_w=0.5 TRAIN.tl_aux_w=0.5
wait
echo "batch3 training done"
for name in r50_tl_aux_js r50_tl_floor_aux_w1 r50_tl_floor15_aux r50_tl_floor_aux_s4; do
  d=$(ls -dt outputs/${name}/*/ 2>/dev/null | head -1)
  [ -z "$d" ] && { echo "MISSING $name"; continue; }
  "$PY" scripts/test_tta.py --checkpoint "${d}swa_f0.5.pth" --views base \
    --dump "dumps/single_${name}" > "logs/${name}_eval.log" 2>&1
  "$PY" scripts/probe_mask_on_object.py --checkpoint "${d}swa_f0.5.pth" --n 96 \
    > "logs/${name}_obj.log" 2>&1
  "$PY" scripts/probe_token_stats.py --checkpoint "${d}swa_f0.5.pth" --n 96 --batch-size 8 \
    > "logs/${name}_tok.log" 2>&1
  echo "eval $name done"
done
echo "batch3 done"
