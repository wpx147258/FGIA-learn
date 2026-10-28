#!/usr/bin/env bash
# 远程第四批（2026-10-29 改版，插在 batch3 之后）：拼接读出 + 部件锚定。
#
# 动机（batch6 的结论）：mean 读出把 S 个部件先平均再分类，与 GAP 同构，
# 正则只能**重新分配**权重、不能**增加**信息；实测富集比最高的 rep_js_w01(1.63×)
# 准确率最差(−3.81)。改成 replace+concat 读出：每个 token 各自投影到 C/S 维再拼起来，
# 总维仍是 C，分类头大小与基线相同，但能对每个部件**分别加权**，部件信息才有通道。
#
# 四个格子 = floor 0.5 下的 2×2（JS × 逐 token CE）：
#   g0 控制臂：都不开（八张图会塌成同一张 = GAP，用来隔离 concat 结构本身）
#   g1 只有锚（aux）
#   g2 只有分化（JS）
#   g3 锚 + 分化（预期基准）
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
launch 0 r50_tl_cat_plain MODEL.tokenlearner=true MODEL.tl_replace=true MODEL.tl_readout=concat \
         MODEL.tl_dropout=0.0 MODEL.tl_ent_mode=floor MODEL.tl_ent_floor=0.5
launch 1 r50_tl_cat_aux MODEL.tokenlearner=true MODEL.tl_replace=true MODEL.tl_readout=concat \
         MODEL.tl_dropout=0.0 MODEL.tl_ent_mode=floor MODEL.tl_ent_floor=0.5 TRAIN.tl_aux_w=0.5
launch 2 r50_tl_cat_js MODEL.tokenlearner=true MODEL.tl_replace=true MODEL.tl_readout=concat \
         MODEL.tl_dropout=0.0 MODEL.tl_ent_mode=floor MODEL.tl_ent_floor=0.5 \
         MODEL.tl_js_w=0.5 MODEL.tl_js_target=0.35 TRAIN.tl_reg_w=0.5
launch 3 r50_tl_cat_js_aux MODEL.tokenlearner=true MODEL.tl_replace=true MODEL.tl_readout=concat \
         MODEL.tl_dropout=0.0 MODEL.tl_ent_mode=floor MODEL.tl_ent_floor=0.5 \
         MODEL.tl_js_w=0.5 MODEL.tl_js_target=0.35 TRAIN.tl_reg_w=0.5 TRAIN.tl_aux_w=0.5
wait
echo "batch4 training done"
for name in r50_tl_cat_plain r50_tl_cat_aux r50_tl_cat_js r50_tl_cat_js_aux; do
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
echo "batch4 done"
