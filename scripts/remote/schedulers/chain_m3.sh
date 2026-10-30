#!/usr/bin/env bash
# 2026-10-30 23:20：三尺度一致性的跨机器/多种子验证（远程恢复后执行）。
#
# 本地 SEED=1 上 `e75m3_s1`（三尺度 1.3/1.6/2.0）= f0.6 89.78 / f0.7 89.85，
# 对 `e75m2_s1`(89.58) 是 +0.22，是目前唯一对**结构改动**有正向响应的结果。
# 但只有一粒，而今天反复验证了单粒不可信，所以需要：
#   S42 —— 与远程已有的 e75_m2(89.68) / base(89.59 f0.6) 直接配对（跨机器独立验证）
#   S2  —— 第三个种子，与 base_s2 同口径
# 另外顺带补评 e100m2（远程失联时它停在 ep98，若 swa 权重已存就评掉）。
cd /root/autodl-tmp/cub-bird || exit 1
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
LOG=/root/chain_m3.log
echo "[$(date '+%F %T')] 启动三尺度验证批" >> "$LOG"

BASE="--config configs/baseline_config.yaml --set DATASET.image_size=448 TRAIN.epochs=75 TRAIN.batch_size=16 TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 TRAIN.cam_margin2=1.6 TRAIN.cam_consistency2=0.15 TRAIN.cam_margin3=2.0 TRAIN.cam_consistency3=0.15 TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3 MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1 TRAIN.swa_start_frac=[0.6,0.7,0.8] DATASET.train_on_val=true"

CUDA_VISIBLE_DEVICES=0 nohup $PY train.py $BASE SEED=42 EXP_NAME=r50_all_e75m3_s42 > logs/r50_all_e75m3_s42_train.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup $PY train.py $BASE SEED=2 EXP_NAME=r50_all_e75m3_s2 > logs/r50_all_e75m3_s2_train.log 2>&1 &
echo "[$(date '+%F %T')] launch e75m3_s42 / e75m3_s2" >> "$LOG"

# 顺带补评 e100m2（失联时停在 ep98）
d=$(ls -dt outputs/r50_all_e75m2_e100/*/ 2>/dev/null | head -1)
if [ -n "$d" ] && ls "$d"*.pth >/dev/null 2>&1; then
  echo "[$(date '+%F %T')] 补评 e100m2" >> "$LOG"
  CUDA_VISIBLE_DEVICES=2 $PY scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1
fi

for _ in $(seq 1 60); do pgrep -f "EXP_NAME=(r50_all_e75m3_s42|r50_all_e75m3_s2)$" >/dev/null 2>&1 && break; sleep 20; done
while pgrep -f "EXP_NAME=(r50_all_e75m3_s42|r50_all_e75m3_s2)$" >/dev/null 2>&1; do sleep 30; done
echo "[$(date '+%F %T')] 训练结束，评测" >> "$LOG"
for n in r50_all_e75m3_s42 r50_all_e75m3_s2; do
  d=$(ls -dt outputs/$n/*/ 2>/dev/null|head -1)
  [ -n "$d" ] && { echo "--- eval $n" >> "$LOG"; $PY scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1; }
done
echo "[$(date '+%F %T')] CHAIN_M3_DONE" >> "$LOG"
