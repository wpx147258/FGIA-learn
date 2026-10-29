#!/usr/bin/env bash
# 2026-10-30 03:00：按已有结论挑出的「还没排、但最可能涨点」的四格。
#
# 依据（全部是本项目自己测出来的）：
#   1. **CAM 裁剪分支是全项目最大的单个机制**：`r50_nocam` 88.21 vs 带分支 89.06 → **+0.85~1.19**。
#      它是唯一被反复验证为正的组件，但只扫过 margin / alpha，**从没扫过它的损失权重**。
#      → 两格：cam_crop 0.2→0.35（更强）、cam_consistency 0.3→0.6（更强的一致性）。
#   2. **输入分辨率一次都没动过**：全程 448（训练 RandomResizedCrop 448，测试 resize 512→裁 448）。
#      细粒度上分辨率是标准增益项，而 stride-8（改特征图尺寸）试过且更差（88.56），
#      说明"改分辨率"和"改下采样"不是一回事，值得单独试。
#      → 两格：基线 @512、最好的 PA 配置 @512。
#
# 排在 TRUE_REPRO_DONE 之后、chain_restore 之前（补回的老实验让位给新实验）。
cd /root/autodl-tmp/cub-bird || exit 1
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
LOG=/root/chain_res.log
for _ in $(seq 1 900); do grep -q "TRUE_REPRO_DONE" /root/chain_true_repro.log 2>/dev/null && break; sleep 60; done
echo "[$(date '+%F %T')] 前置链路结束，启动「分辨率 + CAM 强度」四格" >> "$LOG"

BASE="--config configs/baseline_config.yaml --set DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16 TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3 MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1 TRAIN.swa_start_frac=[0.5,0.6,0.7] DATASET.train_on_val=true"
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
evalbatch () { for n in "$@"; do d=$(ls -dt outputs/$n/*/ 2>/dev/null|head -1)
  [ -n "$d" ] && { echo "--- eval $n" >> "$LOG"; $PY scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1; }; done; }

# 512 的评测显存也更大，EVAL.batch_size 从 64 降到 32（只影响评测吞吐，不改结果）
R512="DATASET.image_size=512 EVAL.batch_size=32"

echo "=== R1：分辨率 512 两格 ===" >> "$LOG"
launch 0 r50_all_r512_base  $R512
launch 1 r50_all_r512_pa4k8 $R512 $PA8
echo "=== R2：CAM 分支强度两格（448）===" >> "$LOG"
launch 2 r50_all_cam035     TRAIN.cam_crop=0.35
launch 3 r50_all_cons06     TRAIN.cam_consistency=0.6
waitnames "EXP_NAME=(r50_all_r512_base|r50_all_r512_pa4k8|r50_all_cam035|r50_all_cons06)$"
evalbatch r50_all_r512_base r50_all_r512_pa4k8 r50_all_cam035 r50_all_cons06
cleanup    r50_all_r512_base r50_all_r512_pa4k8 r50_all_cam035 r50_all_cons06
echo "[$(date '+%F %T')] CHAIN_RES_DONE" >> "$LOG"
