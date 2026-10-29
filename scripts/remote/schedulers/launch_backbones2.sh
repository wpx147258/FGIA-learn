#!/usr/bin/env bash
# 换预训练权重的首批：a1_in1k（rms 0.083）与 tv_in1k（rms 0.423），各跑
# 「注意力最优配置 pa4_k8」+「同权重无注意力控制臂」。lr 按 sqrt(0.711/rms) 缩放：
#   统一用 1e-3，与历史所有实验一致（不混入 lr 变量）

cd /root/autodl-tmp/cub-bird || exit 1
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
COMMON="--config configs/baseline_config.yaml --set DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16 TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 TRAIN.backbone_lr_layers.layer4=0.05 MODEL.dropout=0.1 TRAIN.swa_start_frac=[0.5,0.6,0.7]"
PA="MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=8 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 TRAIN.pa_div_w=1.0 TRAIN.pa_aux_w=0.25 TRAIN.pa_lr=5.0e-4"
launch () { local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY train.py $COMMON "$@" EXP_NAME="$name" > "logs/${name}_train.log" 2>&1 &
  echo "launch $name gpu$gpu pid $!"; }
launch 0 r50_a1_base   MODEL.name=resnet50.a1_in1k TRAIN.lr=1.0e-3
launch 1 r50_a1_pa4k8  MODEL.name=resnet50.a1_in1k TRAIN.lr=1.0e-3 $PA
launch 2 r50_tv_base   MODEL.name=resnet50.tv_in1k TRAIN.lr=1.0e-3
launch 3 r50_tv_pa4k8  MODEL.name=resnet50.tv_in1k TRAIN.lr=1.0e-3 $PA
wait
echo "BACKBONES_DONE"
