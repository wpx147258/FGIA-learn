#!/usr/bin/env bash
# 补齐 OPTIMIZATION_BACKLOG 里的 P0/P1 对齐臂（单视图）：
#   e0 attn_tokens=49（196→49，注意力矩阵小 16 倍）
#   e1 attn_depth=0 控制臂（同参数、无 token 混合）
#   e2 attn_dropout=0（与 PartAttention/GlobalContext 的 dropout 口径对齐）
#   e3 tl_dropout=0（同上，TokenLearner 版）
cd /root/autodl-tmp/cub-bird || exit 1
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
COMMON="--config configs/baseline_config.yaml --set DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16 TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3 MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1 TRAIN.swa_start_frac=[0.5,0.6,0.7]"
wait_gpu() { for _ in $(seq 1 360); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"); [ "$m" -lt 2000 ] && sleep 20 && return 0; sleep 20; done; return 1; }
go() { local gpu=$1 name=$2; shift 2; wait_gpu $gpu || return 1
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY train.py $COMMON "$@" EXP_NAME="$name" > "logs/${name}_train.log" 2>&1 &
  echo "launch $name gpu$gpu pid $!"; }
go 0 r50_attnp_t49_sv MODEL.attn_pool=true MODEL.attn_readout=cls MODEL.attn_depth=1 MODEL.attn_tokens=49 &
go 1 r50_attnp_d0_sv  MODEL.attn_pool=true MODEL.attn_readout=cls MODEL.attn_depth=0 &
go 2 r50_attnp_nodrop MODEL.attn_pool=true MODEL.attn_readout=cls MODEL.attn_depth=1 MODEL.attn_dropout=0.0 &
go 3 r50_tl_nodrop    MODEL.tokenlearner=true MODEL.tl_readout=cls MODEL.tl_attn_depth=1 MODEL.tl_dropout=0.0 &
wait
echo "PHASE_E_DONE"
