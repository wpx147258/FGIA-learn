#!/usr/bin/env bash
# 2026-10-30 09:40：在 `e75m2` 上叠加注意力模块（全训，SEED=42）。
#
# 依据：`e75m2`（75 轮 + 双尺度一致性）是目前唯一有重复证据的正信号——
#   三对同机同种子配对 +0.61 / +0.32 / +0.09，绝对值稳定在 89.6~89.7；
#   而 AttnPool-CLS（89.85）与 PartAttention K=8（89.75）各自在噪声内，
#   且两者与 e75m2 正交（一个改训练长度+损失，一个加模块）。
# 目标：那 0.15 分——若叠加能到 89.8~89.9，就是目前最接近 90 的配置。
#
# 四格：
#   e75m2_attnpcls    = e75m2 + AttnPool-CLS（当前数字最好的模块）
#   e75m2_pa4k8       = e75m2 + PartAttention K=8（主线机制）
#   e75m2_both        = e75m2 + 两者同时（若单独都为正则，叠加是否更稳）
#   e75m2_cam035      = e75m2 + CAM 权重 0.35（CAM 是全项目最大机制，单独 89.70）
#
# 排法：G4 训练一结束（"=== G4" 出现，评测占用 GPU0）就先在 GPU1/2/3 起三格，
# 等 G4 评完（CHAIN_GAP_DONE）再在 GPU0 起第四格，避免和评测抢卡。
cd /root/autodl-tmp/cub-bird || exit 1
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
LOG=/root/chain_stack.log
# ⚠️ 必须等 CHAIN_GAP_DONE（G4 训练+评测+清理全部结束），不能等 "=== G4"——
# 那个标记是 G4 **开始**时打的，等它等于让叠加批和 G4 抢同一批卡（已实测 OOM 一次）。
for _ in $(seq 1 900); do grep -q "CHAIN_GAP_DONE" /root/chain_gap.log 2>/dev/null && break; sleep 30; done
echo "[$(date '+%F %T')] G4 全部结束，启动叠加批" >> "$LOG"

BASE="--config configs/baseline_config.yaml --set DATASET.image_size=448 TRAIN.epochs=75 TRAIN.batch_size=16 TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 TRAIN.cam_margin2=1.6 TRAIN.cam_consistency2=0.15 TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3 MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1 TRAIN.swa_start_frac=[0.6,0.7,0.8] DATASET.train_on_val=true"
AT="MODEL.attn_pool=true MODEL.attn_readout=cls MODEL.attn_depth=1 MODEL.attn_dropout=0.0"
PA8="MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=8 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 TRAIN.pa_div_w=1.0 TRAIN.pa_aux_w=0.5 TRAIN.pa_lr=5.0e-4"

KEEP="r50_all_pa4k8 r50_all_base r50_all_attnpcls"
cleanup () { for n in "$@"; do
  cnt=0; for c in best last swa_f05 swa_f06 swa_f07; do
    [ -f "dumps/single_${n}_${c}_probs.npy" ] && cnt=$((cnt+1)); done
  case " $KEEP " in *" $n "*) echo "[cleanup] keep $n" >> "$LOG"; continue;; esac
  if [ "$cnt" -eq 5 ]; then rm -f outputs/$n/*/*.pth; echo "[cleanup] $n 5/5 dump 齐 -> 删 ckpt" >> "$LOG";
  else echo "[cleanup] $n 只有 $cnt/5 dump，保留 ckpt" >> "$LOG"; fi
done; }
launch () { local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY train.py $BASE "$@" EXP_NAME="$name" > "logs/${name}_train.log" 2>&1 &
  echo "[$(date '+%F %T')] launch $name gpu$gpu" >> "$LOG"; }
waitnames () { local pat="$1"
  for _ in $(seq 1 60); do pgrep -f "$pat" >/dev/null 2>&1 && break; sleep 20; done
  while pgrep -f "$pat" >/dev/null 2>&1; do sleep 30; done; }

launch 0 r50_all_e75m2_cam035   TRAIN.cam_crop=0.35
launch 1 r50_all_e75m2_attnpcls $AT
launch 2 r50_all_e75m2_pa4k8    $PA8
launch 3 r50_all_e75m2_both     $AT $PA8

waitnames "EXP_NAME=(r50_all_e75m2_attnpcls|r50_all_e75m2_pa4k8|r50_all_e75m2_both|r50_all_e75m2_cam035)$"
echo "[$(date '+%F %T')] 叠加四格训练结束，评测" >> "$LOG"
for n in r50_all_e75m2_attnpcls r50_all_e75m2_pa4k8 r50_all_e75m2_both r50_all_e75m2_cam035; do
  d=$(ls -dt outputs/$n/*/ 2>/dev/null|head -1)
  [ -n "$d" ] && { echo "--- eval $n" >> "$LOG"; $PY scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1; }
done
cleanup r50_all_e75m2_attnpcls r50_all_e75m2_pa4k8 r50_all_e75m2_both r50_all_e75m2_cam035
echo "[$(date '+%F %T')] CHAIN_STACK_DONE" >> "$LOG"
