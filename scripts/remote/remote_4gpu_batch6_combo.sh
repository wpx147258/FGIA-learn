#!/usr/bin/env bash
# 远程第六批：特权裁剪的组合与收尾（batch5 出分后使用）。
#
# 逻辑：单通道（CAM 框 / GT 头）都已经只能做到持平，本批测**叠加与耦合**：
#   g0 cam_head_w05_pc : CAM 框 448 + GT 头 224 + 部位一致性 0.3（新加的约束）
#   g1 gth_pad08_wing  : GT 头（pad 0.8，448）+ GT 翅 224
#   g2 gth_pad08_attn  : GT 头 + AttnPool(CLS,d1)（两个各自持平机制的组合）
#   g3 cam_head_w05_c04: 同 g0 的输入，但把总一致性权重调到 0.4（调参格）
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
launch 0 r50_cam_head_w05_pc TRAIN.cam_teacher=cam TRAIN.cam_part_views=head \
         TRAIN.cam_part_w=0.5 TRAIN.cam_part_cons=0.3
launch 1 r50_gth_pad08_wing TRAIN.cam_teacher=gt_head TRAIN.cam_crop_hard_label=true \
         TRAIN.part_pad=0.8 TRAIN.cam_part_views=wing TRAIN.cam_part_w=0.3
launch 2 r50_gth_pad08_attn TRAIN.cam_teacher=gt_head TRAIN.cam_crop_hard_label=true \
         TRAIN.part_pad=0.8 MODEL.attn_pool=true MODEL.attn_readout=cls MODEL.attn_depth=1
launch 3 r50_cam_head_w05_c04 TRAIN.cam_teacher=cam TRAIN.cam_part_views=head \
         TRAIN.cam_part_w=0.5 TRAIN.cam_consistency=0.4
wait
echo "batch6 training done"
for name in r50_cam_head_w05_pc r50_gth_pad08_wing r50_gth_pad08_attn r50_cam_head_w05_c04; do
  d=$(ls -dt outputs/${name}/*/ 2>/dev/null | head -1)
  [ -z "$d" ] && { echo "MISSING $name"; continue; }
  "$PY" scripts/test_tta.py --checkpoint "${d}swa_f0.5.pth" --views base \
    --dump "dumps/single_${name}" > "logs/${name}_eval.log" 2>&1
  echo "eval $name done"
done
echo "batch6 done"
