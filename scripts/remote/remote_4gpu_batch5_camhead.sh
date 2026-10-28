#!/usr/bin/env bash
# 远程第五批（2026-10-29 改版 v2）：在**已证明有效**的 CAM 裁剪之上，再加一路 GT 部位视图。
#
# 为什么这样排：cam_crop（CAM 框裁出来重编码）是本项目唯一稳定正项（+0.79），
# 而 batch4 测的是「用 GT 头**替换** CAM 框」。这一批测的是**叠加**：
# 主图 448 + CAM 框 448（原配方不动）+ 一路 GT 部位 224（新增）。
# 224 是为了显存：三路 448 会顶到 12GB，224 的额外视图只多约 0.9GB。
#
# 四格：
#   g0 cam_head_w03  CAM 裁剪 + GT 头（224），辅助 CE 0.3
#   g1 cam_head_w05  同上，0.5
#   g2 gth_wing      gt_head 教师（替换 CAM）+ GT 翅（224）0.3
#   g3 cam_wing_w03  CAM 裁剪 + GT 翅（224）0.3
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
  TRAIN.cam_part_size=224
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
launch 0 r50_cam_head_w03 TRAIN.cam_teacher=cam TRAIN.cam_part_views=head TRAIN.cam_part_w=0.3
launch 1 r50_cam_head_w05 TRAIN.cam_teacher=cam TRAIN.cam_part_views=head TRAIN.cam_part_w=0.5
launch 2 r50_gth_wing     TRAIN.cam_teacher=gt_head TRAIN.cam_crop_hard_label=true \
         TRAIN.cam_part_views=wing TRAIN.cam_part_w=0.3
launch 3 r50_cam_wing_w03 TRAIN.cam_teacher=cam TRAIN.cam_part_views=wing TRAIN.cam_part_w=0.3
wait
echo "batch5 training done"
for name in r50_cam_head_w03 r50_cam_head_w05 r50_gth_wing r50_cam_wing_w03; do
  d=$(ls -dt outputs/${name}/*/ 2>/dev/null | head -1)
  [ -z "$d" ] && { echo "MISSING $name"; continue; }
  "$PY" scripts/test_tta.py --checkpoint "${d}swa_f0.5.pth" --views base \
    --dump "dumps/single_${name}" > "logs/${name}_eval.log" 2>&1
  echo "eval $name done"
done
echo "batch5 done"
