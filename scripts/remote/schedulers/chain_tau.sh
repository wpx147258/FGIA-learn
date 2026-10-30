#!/usr/bin/env bash
# 2026-10-30 18:55：τ（soft partition 温度）扫描的扩展批。
#
# 依据：K=4 上 τ 的效应是**单调**的——0.05→89.30、0.10→89.54、**0.20→89.73**。
# τ 越大分区越软、部件之间允许部分重叠，正是"眼睛/头部天然重叠"的机制实现；
# 而彻底取消分区（compete=false）会塌成同一张掩码（89.25/89.07）。
#
# 这一批在 **K=8**（最好的一档）上继续往上扫，全部 SEED=42，
# 直接与同机的 `pa4k8`（K=8, τ=0.1）= **89.78** 配对比较。
cd /root/autodl-tmp/cub-bird || exit 1
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
LOG=/root/chain_tau.log
for _ in $(seq 1 900); do grep -q "CHAIN_RESTORE_DONE" /root/chain_restore.log 2>/dev/null && break; sleep 60; done
echo "[$(date '+%F %T')] 补回批结束，启动 τ 扩展四格" >> "$LOG"

BASE="--config configs/baseline_config.yaml --set DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16 TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3 MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1 TRAIN.swa_start_frac=[0.5,0.6,0.7] DATASET.train_on_val=true"
PA8="MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=8 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true TRAIN.pa_div_w=1.0 TRAIN.pa_aux_w=0.5 TRAIN.pa_lr=5.0e-4"

KEEP="r50_all_pa4k8 r50_all_base r50_all_attnpcls"
cleanup () { for n in "$@"; do
  cnt=0; for c in best last swa_f05 swa_f06 swa_f07; do
    [ -f "dumps/single_${n}_${c}_probs.npy" ] && cnt=$((cnt+1)); done
  case " $KEEP " in *" $n "*) echo "[cleanup] keep $n" >> "$LOG"; continue;; esac
  if [ "$cnt" -eq 5 ]; then rm -f outputs/$n/*/*.pth; echo "[cleanup] $n 5/5 dump 齐 -> 删 ckpt" >> "$LOG"; fi
done; }
launch () { local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY train.py $BASE "$@" EXP_NAME="$name" > "logs/${name}_train.log" 2>&1 &
  echo "[$(date '+%F %T')] launch $name gpu$gpu" >> "$LOG"; }
waitnames () { local pat="$1"
  for _ in $(seq 1 60); do pgrep -f "$pat" >/dev/null 2>&1 && break; sleep 20; done
  while pgrep -f "$pat" >/dev/null 2>&1; do sleep 30; done; }
evalbatch () { for n in "$@"; do d=$(ls -dt outputs/$n/*/ 2>/dev/null|head -1)
  [ -n "$d" ] && { echo "--- eval $n" >> "$LOG"; $PY scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1; }; done; }

launch 0 r50_all_pa8_tau02 $PA8 MODEL.pa_tau=0.2
launch 1 r50_all_pa8_tau03 $PA8 MODEL.pa_tau=0.3
launch 2 r50_all_pa8_tau05 $PA8 MODEL.pa_tau=0.5
launch 3 r50_all_pa8_tau08 $PA8 MODEL.pa_tau=0.8
waitnames "EXP_NAME=(r50_all_pa8_tau02|r50_all_pa8_tau03|r50_all_pa8_tau05|r50_all_pa8_tau08)$"
evalbatch r50_all_pa8_tau02 r50_all_pa8_tau03 r50_all_pa8_tau05 r50_all_pa8_tau08
cleanup    r50_all_pa8_tau02 r50_all_pa8_tau03 r50_all_pa8_tau05 r50_all_pa8_tau08
echo "[$(date '+%F %T')] CHAIN_TAU_DONE" >> "$LOG"
