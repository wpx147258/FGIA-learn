#!/usr/bin/env bash
# 补最关键的缺口：全训 + 更长训练（75/100 epoch）。排在 PA 调优批（chain_pa）之后。
# 注意 long75（5,395）当年的配方里**同时**改了两件事：75 epoch + 双尺度一致性
# （cam_margin2=1.6, cam_consistency2=0.15）。这里拆开：一格拉满配方的复刻、一格只加长度。
#
# 2026-10-30 01:45 修订：
#   - 前置条件从 CHAIN_CLS_DONE 改成 CHAIN_PA_DONE（PA 调优批插在前面）；
#   - `wait` 改成按 EXP_NAME 轮询（旧版实测会在 nohup 下不等，导致下一批抢卡 OOM）。
cd /root/autodl-tmp/cub-bird || exit 1
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
LOG=/root/chain_long.log
for _ in $(seq 1 500); do
  grep -q "CHAIN_PA_DONE" /root/chain_pa.log 2>/dev/null && break
  sleep 60
done
echo "[$(date '+%F %T')] 前置批次结束，启动长训练四格" >> "$LOG"
BASE="--config configs/baseline_config.yaml --set DATASET.image_size=448 TRAIN.batch_size=16 TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3 MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1 DATASET.train_on_val=true"
SWA="TRAIN.swa_start_frac=[0.6,0.7,0.8]"
# 评测完之后删掉本批 checkpoint，防止再次写满 70G 数据盘（2026-10-30 02:30 满盘事故）
KEEP="r50_all_pa4k8 r50_all_base r50_all_attnpcls"
cleanup () { for n in "$@"; do
  cnt=0; for c in best last swa_f05 swa_f06 swa_f07; do
    [ -f "dumps/single_${n}_${c}_probs.npy" ] && cnt=$((cnt+1)); done
  case " $KEEP " in *" $n "*) echo "[cleanup] keep $n" >> "$LOG"; continue;; esac
  if [ "$cnt" -eq 5 ]; then rm -f outputs/$n/*/*.pth; echo "[cleanup] $n 5/5 dump 齐 -> 删 ckpt" >> "$LOG";
  else echo "[cleanup] $n 只有 $cnt/5 dump，保留 ckpt" >> "$LOG"; fi
done; }

PAT="EXP_NAME=(r50_all_e75|r50_all_e75_m2|r50_all_e100|r50_all_attnpcls_e75)$"
launch () { local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY train.py $BASE "$@" EXP_NAME="$name" > "logs/${name}_train.log" 2>&1 &
  echo "[$(date '+%F %T')] launch $name gpu$gpu" >> "$LOG"; }
launch 0 r50_all_e75     TRAIN.epochs=75 $SWA
launch 1 r50_all_e75_m2  TRAIN.epochs=75 $SWA TRAIN.cam_margin2=1.6 TRAIN.cam_consistency2=0.15
launch 2 r50_all_e100    TRAIN.epochs=100 $SWA
launch 3 r50_all_attnpcls_e75 TRAIN.epochs=75 $SWA MODEL.attn_pool=true MODEL.attn_readout=cls MODEL.attn_depth=1 MODEL.attn_dropout=0.0
for _ in $(seq 1 60); do pgrep -f "$PAT" >/dev/null 2>&1 && break; sleep 20; done
while pgrep -f "$PAT" >/dev/null 2>&1; do sleep 30; done
echo "[$(date '+%F %T')] 长训练四格结束，评测" >> "$LOG"
for n in r50_all_e75 r50_all_e75_m2 r50_all_e100 r50_all_attnpcls_e75; do
  d=$(ls -dt outputs/$n/*/ 2>/dev/null|head -1)
  [ -n "$d" ] && { echo "--- eval $n" >> "$LOG"; $PY scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1; }
done
cleanup r50_all_e75 r50_all_e75_m2 r50_all_e100 r50_all_attnpcls_e75
echo "[$(date '+%F %T')] CHAIN_LONG_DONE" >> "$LOG"
