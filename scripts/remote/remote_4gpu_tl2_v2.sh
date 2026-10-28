#!/usr/bin/env bash
# 远程 TL 第二轮（v2，2026-10-29）：在「替换出口 + 单侧熵下限」这个基准上扫参数。
#
# 为什么改了这一版：batch6 前 10 个 epoch 已经看出，replace 出口 + 双侧熵目标
# (ent−2.0)² 在初始化（均匀权重熵 = ln196 ≈ 5.28）时惩罚 ≈10.8，直接改写分类
# 输入，ep3 只有 52% 而基线 79%。所以「双侧 target」不适合 replace 出口，
# 本轮的四个格子全部用单侧 floor，只动一个量：
#   g0 基准      floor 0.5 + JS 0.35
#   g1 弱 JS     floor 0.5 + JS 0.15   （允许图间重叠更多）
#   g2 强 floor  floor 1.5 + JS 0.35   （部件必须是「区域」而不是单点）
#   g3 少 token  S=4 + floor 0.5 + JS 0.35
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
launch 0 r50_tl_rep_floor MODEL.tokenlearner=true MODEL.tl_replace=true MODEL.tl_dropout=0.0 \
         MODEL.tl_ent_mode=floor MODEL.tl_ent_floor=0.5 MODEL.tl_js_w=0.5 MODEL.tl_js_target=0.35 \
         TRAIN.tl_reg_w=0.5
launch 1 r50_tl_rep_floor_j015 MODEL.tokenlearner=true MODEL.tl_replace=true MODEL.tl_dropout=0.0 \
         MODEL.tl_ent_mode=floor MODEL.tl_ent_floor=0.5 MODEL.tl_js_w=0.5 MODEL.tl_js_target=0.15 \
         TRAIN.tl_reg_w=0.5
launch 2 r50_tl_rep_floor15 MODEL.tokenlearner=true MODEL.tl_replace=true MODEL.tl_dropout=0.0 \
         MODEL.tl_ent_mode=floor MODEL.tl_ent_floor=1.5 MODEL.tl_js_w=0.5 MODEL.tl_js_target=0.35 \
         TRAIN.tl_reg_w=0.5
launch 3 r50_tl_rep_floor_s4 MODEL.tokenlearner=true MODEL.tl_replace=true MODEL.tl_dropout=0.0 \
         MODEL.tl_tokens=4 MODEL.tl_ent_mode=floor MODEL.tl_ent_floor=0.5 MODEL.tl_js_w=0.5 \
         MODEL.tl_js_target=0.35 TRAIN.tl_reg_w=0.5
wait
echo "tl2 training done"
for name in r50_tl_rep_floor r50_tl_rep_floor_j015 r50_tl_rep_floor15 r50_tl_rep_floor_s4; do
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
echo "tl2 done"
