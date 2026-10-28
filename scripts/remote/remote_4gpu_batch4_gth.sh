#!/usr/bin/env bash
# 远程第四批（2026-10-29 改版 v2）：训练期特权裁剪的多部位版本。
#
# 独立审查的结论：在合法机制里唯一「增加信息」的通道是裁剪重编码（cam_crop +0.79 是
# 本项目唯一稳定正项；NTS-Net K=4 vs K=0 的 +2.2 全部来自多区域重编码），而注意力
# 重加权与 GAP 同构。所以这一轮把「裁什么」从 CAM 框换成 GT 关键点框，并增加部位数。
#
# 四个格子（都在 baseline 配方上；主图仍是原图 + CAM 一致性那套）：
#   g0 gt_head            —— 只有头部（与本地同名实验构成两个 seed）
#   g1 gt_head_body       —— 头 + 身
#   g2 gt_head_wing       —— 头 + 翅
#   g3 gt_head_wing_tail  —— 头 + 翅 + 尾（前向次数最多，决定整批时长）
#
# 训练时多一次前向是允许的；测试路径完全不经过这些分支（单图单前向）。
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
  TRAIN.cam_teacher=gt_head TRAIN.cam_crop_hard_label=true
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
launch 0 r50_gth
launch 1 r50_gth_body     TRAIN.cam_part_views=body   TRAIN.cam_part_w=0.3
launch 2 r50_gth_wing     TRAIN.cam_part_views=wing   TRAIN.cam_part_w=0.3
launch 3 r50_gth_wing_tail TRAIN.cam_part_views=wing,tail TRAIN.cam_part_w=0.3
wait
echo "batch4 training done"
for name in r50_gth r50_gth_body r50_gth_wing r50_gth_wing_tail; do
  d=$(ls -dt outputs/${name}/*/ 2>/dev/null | head -1)
  [ -z "$d" ] && { echo "MISSING $name"; continue; }
  "$PY" scripts/test_tta.py --checkpoint "${d}swa_f0.5.pth" --views base \
    --dump "dumps/single_${name}" > "logs/${name}_eval.log" 2>&1
  echo "eval $name done"
done
echo "batch4 done"
