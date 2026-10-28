#!/usr/bin/env bash
# 远程第四批（2026-10-29 改版 v3）：训练期特权裁剪（GT 头部图）的四种配置。
#
# 为什么先只做「头」：显存。基线本身已经是 2 次前向（主图 + CAM/头部裁剪），实测
# 约 8.6GB/卡（12GB）。再加任一路 448 的辅助视图都会逼近上限，多部位版本要先把
# 批大小或梯度累积改掉（见 AUTO_STATE 的记录），先用零风险的单视图把机制跑清楚。
#
# 与本地 queueRR 的四格同名同参 —— 两边的结果合起来就是「同一配置的两个 seed」，
# 正好检验 seed 方差（前面 pa4 的教训）。
#   g0 hl      : 硬标签（头部图来自原图，mixup 软目标是错误监督）
#   g1 soft    : 软目标（量化硬标签这一项值多少）
#   g2 c05     : 一致性 0.5（基线的 0.3 是给 CAM 框调的）
#   g3 nocons  : 一致性 0（只保留裁剪分支自己的 CE）
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
launch 0 r50_gth_hl     TRAIN.cam_teacher=gt_head TRAIN.cam_crop_hard_label=true
launch 1 r50_gth_soft   TRAIN.cam_teacher=gt_head
launch 2 r50_gth_c05    TRAIN.cam_teacher=gt_head TRAIN.cam_crop_hard_label=true \
         TRAIN.cam_consistency=0.5
launch 3 r50_gth_nocons TRAIN.cam_teacher=gt_head TRAIN.cam_crop_hard_label=true \
         TRAIN.cam_consistency=0.0
wait
echo "batch4 training done"
for name in r50_gth_hl r50_gth_soft r50_gth_c05 r50_gth_nocons; do
  d=$(ls -dt outputs/${name}/*/ 2>/dev/null | head -1)
  [ -z "$d" ] && { echo "MISSING $name"; continue; }
  "$PY" scripts/test_tta.py --checkpoint "${d}swa_f0.5.pth" --views base \
    --dump "dumps/single_${name}" > "logs/${name}_eval.log" 2>&1
  echo "eval $name done"
done
echo "batch4 done"
