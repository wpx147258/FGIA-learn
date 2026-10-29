#!/usr/bin/env bash
# 等 GPU1 / GPU3 空出来后，各跑一个 pa4_k8 的 seed（SEED=1 / 2），
# 用来判定 89.21 是机制还是种子运气（K=4 的同一机制三 seed 是 89.14/88.71/88.63）。
cd /root/autodl-tmp/cub-bird || exit 1
export HF_ENDPOINT=https://hf-mirror.com
C="--config configs/baseline_config.yaml --set DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16 TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3 MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1 TRAIN.swa_start_frac=[0.5,0.6,0.7] MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=8 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 MODEL.pa_depth=1 MODEL.pa_div_w=1.0 TRAIN.pa_aux_w=0.25 TRAIN.pa_lr=5.0e-4"
wait_gpu () {  # $1 = gpu index
  for _ in $(seq 1 240); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1")
    [ "$m" -lt 2000 ] && sleep 20 && return 0
    sleep 20
  done
  return 1
}
if wait_gpu 1; then
  CUDA_VISIBLE_DEVICES=1 nohup /root/miniconda3/bin/python train.py $C SEED=1 EXP_NAME=r50_pa4_k8_s1 > logs/r50_pa4_k8_s1_train.log 2>&1 &
  echo "k8_s1 $!"
fi
if wait_gpu 3; then
  CUDA_VISIBLE_DEVICES=3 nohup /root/miniconda3/bin/python train.py $C SEED=2 EXP_NAME=r50_pa4_k8_s2 > logs/r50_pa4_k8_s2_train.log 2>&1 &
  echo "k8_s2 $!"
fi
