#!/usr/bin/env bash
# 补回 10/27 原计划里从未执行的四格：
#   g0 r50_gc        —— layer3→layer4 之间的全局自注意力（H2 的核心）
#   g1 r50_gc_ctrl   —— 参数对齐、去掉 token 混合的控制臂（H2 的判定要求）
#   g2 r50_clip_gap  —— CLIP RN50 + GAP
#   g3 r50_clip_ap   —— CLIP RN50 + 预训练的注意力池化（与 g2 同权重，只差池化头）
cd /root/autodl-tmp/cub-bird || exit 1
export HF_ENDPOINT=https://hf-mirror.com
BASE="--config configs/baseline_config.yaml --set DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16 TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3 MODEL.dropout=0.1 TRAIN.swa_start_frac=[0.5,0.6,0.7]"
launch_when_free () {  # $1=gpu $2=name $3...=extra args
  local gpu=$1 name=$2; shift 2
  for _ in $(seq 1 240); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
    if [ "$m" -lt 2000 ]; then
      sleep 25
      m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
      if [ "$m" -lt 2000 ]; then
        CUDA_VISIBLE_DEVICES=$gpu nohup /root/miniconda3/bin/python train.py $BASE "$@" EXP_NAME="$name" > "logs/${name}_train.log" 2>&1 &
        echo "launched $name on GPU$gpu pid $!"
        return 0
      fi
    fi
    sleep 20
  done
}
launch_when_free 0 r50_gc       MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.global_ctx=true TRAIN.gc_lr=5.0e-4 &
launch_when_free 1 r50_gc_ctrl  MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.global_ctx=true MODEL.gc_depth=0 TRAIN.gc_lr=5.0e-4 &
launch_when_free 2 r50_clip_gap MODEL.name=resnet50_clip_gap.openai DATASET.crop_pct=0.875 &
launch_when_free 3 r50_clip_ap  MODEL.name=resnet50_clip.openai     DATASET.crop_pct=0.875 &
wait
echo "plan round launched"
