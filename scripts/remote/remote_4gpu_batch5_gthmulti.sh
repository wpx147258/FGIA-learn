#!/usr/bin/env bash
# 远程第五批（预备）：GT 特权裁剪的多部位版（额外视图走 224，省激活）。
#
# 只在 batch4（单视图 gt_head 四配置）显示正收益时才值得跑：
# 审查结论是「多区域重编码是唯一增加信息的合法通道」，但单视图都没涨的话，
# 多视图大概率只是把同样的信号重复一遍。
#
# 四格：
#   g0 gth_wing    头 + 翅
#   g1 gth_wt      头 + 翅 + 尾
#   g2 gth_wb      头 + 身
#   g3 gth_wt_w05  头 + 翅 + 尾，辅助 CE 权重 0.5（默认 0.3）
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
  TRAIN.cam_teacher=gt_head TRAIN.cam_crop_hard_label=true TRAIN.cam_part_size=224
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
launch 0 r50_gth_wing   TRAIN.cam_part_views=wing   TRAIN.cam_part_w=0.3
launch 1 r50_gth_wt     TRAIN.cam_part_views=wing,tail TRAIN.cam_part_w=0.3
launch 2 r50_gth_wb     TRAIN.cam_part_views=body   TRAIN.cam_part_w=0.3
launch 3 r50_gth_wt_w05 TRAIN.cam_part_views=wing,tail TRAIN.cam_part_w=0.5
wait
echo "batch5 training done"
for name in r50_gth_wing r50_gth_wt r50_gth_wb r50_gth_wt_w05; do
  d=$(ls -dt outputs/${name}/*/ 2>/dev/null | head -1)
  [ -z "$d" ] && { echo "MISSING $name"; continue; }
  "$PY" scripts/test_tta.py --checkpoint "${d}swa_f0.5.pth" --views base \
    --dump "dumps/single_${name}" > "logs/${name}_eval.log" 2>&1
  echo "eval $name done"
done
echo "batch5 done"
