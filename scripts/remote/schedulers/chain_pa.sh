#!/usr/bin/env bash
# PA 调优批（全训协议）：等 CLS 三格结束 → 跑四格 → 评测。
#
# 2026-10-30 02:05 按新数据重排。补评出来的两格改写了 PA 的现状：
#   all_pa4k8 (K=8, compete, 无 JS) = 89.75   ← PA 目前最好，且与全训 4 粒均值 89.80 持平
#   all_pa4   (K=4, compete, 无 JS) = 89.58
#   all_pa4js (K=4, compete, JS)    = 89.13   → **JS 在 K=4 上 -0.45**
#   all_sig   (K=4, 无 compete)     = 89.02   → **compete 在 K=4 上 +0.56**
# 也就是说：把 JS 固定住之后，"compete 太强"的假设不成立；真正有效的是 K 和去掉 JS。
# 所以这批四格换成围绕最好配置（K=8 / compete / 无 JS）做：
#   1. parand8   —— 固定随机掩码对照（决定性：掩码到底学没学到东西）
#   2. pa4k16    —— K 的趋势 4→8 是 +0.17，继续往上试
#   3. pa4k8_s7  —— 89.75 的第二个种子（单粒差 0.14 不能算增益）
#   4. pa4k8_l3  —— 检测器挪到 layer3（28×28），resolution 假设
# 掩码平滑的真正复测放在 chain_gap 的 G3（pa4js_blur_v2）。
cd /root/autodl-tmp/cub-bird || exit 1
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
LOG=/root/chain_pa.log
for _ in $(seq 1 400); do
  grep -q "CHAIN_CLS_DONE" /root/chain_cls.log 2>/dev/null && break
  sleep 60
done
echo "[$(date '+%F %T')] CLS 批次结束，启动 PA 调优四格" >> "$LOG"
COMMON="--config configs/baseline_config.yaml --set DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16 TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3 MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1 TRAIN.swa_start_frac=[0.5,0.6,0.7] DATASET.train_on_val=true"
PA8="MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=8 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 TRAIN.pa_div_w=1.0 TRAIN.pa_aux_w=0.5 TRAIN.pa_lr=5.0e-4"
launch () { local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY train.py $COMMON "$@" EXP_NAME="$name" > "logs/${name}_train.log" 2>&1 &
  echo "[$(date '+%F %T')] launch $name gpu$gpu" >> "$LOG"; }
waitnames () { local pat="$1"
  for _ in $(seq 1 60); do pgrep -f "$pat" >/dev/null 2>&1 && break; sleep 20; done
  while pgrep -f "$pat" >/dev/null 2>&1; do sleep 30; done; }

launch 0 r50_all_parand8   $PA8 MODEL.pa_random_masks=true
launch 1 r50_all_pa4k16    $PA8 MODEL.pa_parts=16
launch 2 r50_all_pa4k8_s7  $PA8 SEED=7
launch 3 r50_all_pa4k8_l3  $PA8 MODEL.pa_from=layer3

waitnames "EXP_NAME=(r50_all_parand8|r50_all_pa4k16|r50_all_pa4k8_s7|r50_all_pa4k8_l3)$"
echo "[$(date '+%F %T')] PA 四格训练结束，评测" >> "$LOG"
for n in r50_all_parand8 r50_all_pa4k16 r50_all_pa4k8_s7 r50_all_pa4k8_l3; do
  d=$(ls -dt outputs/$n/*/ 2>/dev/null|head -1)
  [ -n "$d" ] && { echo "--- eval $n" >> "$LOG"; $PY scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1; }
done
echo "[$(date '+%F %T')] CHAIN_PA_DONE" >> "$LOG"
