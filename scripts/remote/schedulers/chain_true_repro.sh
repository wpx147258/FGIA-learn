#!/usr/bin/env bash
# 精确复现原 pa4_js：同机器（3080Ti）+ 同种子 42 + **同 std 自由度（correction=1，分母 N−1=195）**。
# 这是唯一能判定「89.91 是代码差异还是硬件/随机性」的一格。
cd /root/autodl-tmp/cub-bird || exit 1
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
LOG=/root/chain_true_repro.log
# 叠加批（chain_stack）在 G4 之后占用 GPU1/2/3，等它全部跑完再起 true_repro
for _ in $(seq 1 900); do grep -q "CHAIN_STACK_DONE" /root/chain_stack.log 2>/dev/null && break; sleep 60; done
echo "[$(date '+%F %T')] 前置批次结束，启动精确复现" >> "$LOG"
BASE="--config configs/baseline_config.yaml --set DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16 TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3 MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1 TRAIN.swa_start_frac=[0.5,0.6,0.7] MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=4 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 MODEL.pa_js_w=0.5 MODEL.pa_js_target=0.35 TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0 MODEL.pa_std_correction=1"
CUDA_VISIBLE_DEVICES=0 nohup $PY train.py $BASE SEED=42 EXP_NAME=r50_pa4js_c1 > logs/r50_pa4js_c1_train.log 2>&1 &
wait
d=$(ls -dt outputs/r50_pa4js_c1/*/ 2>/dev/null | head -1)
$PY scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] TRUE_REPRO_DONE" >> "$LOG"
