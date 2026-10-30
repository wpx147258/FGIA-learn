#!/usr/bin/env bash
# 2026-10-30 20:20：围绕**唯一被验证有效的成分**做下一步，并顺带复核一个可疑数据点。
#
# 已验证：e75m2（75 轮 + 双尺度一致性）四对配对合并 p=0.015，均值 +0.36。
# 拆解：只加长度 −0.08、只加双尺度 −0.19、两者合并才 +0.28~+0.66（交互效应）。
# 而且一致性**权重**已饱和（0.15→0.30 只 +0.05，p=0.88）。
# 那么下一个没扫过的是**两个尺度的比例本身**（margin2=1.6 是当初拍的值）：
#   m115 → 第二尺度更紧（1.15），两尺度差异小
#   m20  → 第二尺度更松（2.0），两尺度差异大
# 另外补一格 e100m2：长度继续加、且带着双尺度（e100 单独是 89.42，但它没有 m2）。
# 第四格复核 K=4/τ=0.2 那个 89.73 —— 它是 K=4 上唯一的高点，换种子看是否成立。
cd /root/autodl-tmp/cub-bird || exit 1
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
LOG=/root/chain_scale.log
echo "[$(date '+%F %T')] 启动尺度选择批" >> "$LOG"

BASE="--config configs/baseline_config.yaml --set DATASET.image_size=448 TRAIN.epochs=75 TRAIN.batch_size=16 TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 TRAIN.cam_margin2=1.6 TRAIN.cam_consistency2=0.15 TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3 MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1 TRAIN.swa_start_frac=[0.6,0.7,0.8] DATASET.train_on_val=true"
BASE50="--config configs/baseline_config.yaml --set DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16 TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3 MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1 TRAIN.swa_start_frac=[0.5,0.6,0.7] DATASET.train_on_val=true"
PA4="MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=4 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.2 TRAIN.pa_div_w=1.0 TRAIN.pa_aux_w=0.5 TRAIN.pa_lr=5.0e-4"

KEEP="r50_all_pa4k8 r50_all_base r50_all_attnpcls"
cleanup () { for n in "$@"; do
  cnt=0; for c in best last swa_f05 swa_f06 swa_f07; do
    [ -f "dumps/single_${n}_${c}_probs.npy" ] && cnt=$((cnt+1)); done
  case " $KEEP " in *" $n "*) echo "[cleanup] keep $n" >> "$LOG"; continue;; esac
  if [ "$cnt" -eq 5 ]; then rm -f outputs/$n/*/*.pth; echo "[cleanup] $n 5/5 dump 齐" >> "$LOG"; fi
done; }
launch () { local gpu=$1 name=$2; local base=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY train.py $base "$@" EXP_NAME="$name" > "logs/${name}_train.log" 2>&1 &
  echo "[$(date '+%F %T')] launch $name gpu$gpu" >> "$LOG"; }
waitnames () { local pat="$1"
  for _ in $(seq 1 60); do pgrep -f "$pat" >/dev/null 2>&1 && break; sleep 20; done
  while pgrep -f "$pat" >/dev/null 2>&1; do sleep 30; done; }
evalbatch () { for n in "$@"; do d=$(ls -dt outputs/$n/*/ 2>/dev/null|head -1)
  [ -n "$d" ] && { echo "--- eval $n" >> "$LOG"; $PY scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1; }; done; }

launch 0 r50_all_e75m2_m115 "$BASE" TRAIN.cam_margin2=1.15
launch 1 r50_all_e75m2_m20  "$BASE" TRAIN.cam_margin2=2.0
launch 2 r50_all_e75m2_e100 "$BASE" TRAIN.epochs=100
launch 3 r50_all_tau02_k4_s7 "$BASE50" $PA4 SEED=7
waitnames "EXP_NAME=(r50_all_e75m2_m115|r50_all_e75m2_m20|r50_all_e75m2_e100|r50_all_tau02_k4_s7)$"
evalbatch r50_all_e75m2_m115 r50_all_e75m2_m20 r50_all_e75m2_e100 r50_all_tau02_k4_s7
cleanup    r50_all_e75m2_m115 r50_all_e75m2_m20 r50_all_e75m2_e100 r50_all_tau02_k4_s7
echo "[$(date '+%F %T')] CHAIN_SCALE_DONE" >> "$LOG"
