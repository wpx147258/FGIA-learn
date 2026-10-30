#!/usr/bin/env bash
# 2026-10-30 08:40：最后一个未被排除的机制实验——用**真部件**去重新编码。
#
# 依据（PART_ATTENTION_FINDINGS §22 的决定性探针）：
#   逐部件 logits 与主分支预测一致率 95.56%，在主分支出错的那 466 张上只有 16.7% 能猜对
#   （低于随机 20%）。也就是说池化出来的部件证据**与主干同源**，用它与否都一样。
#
#   要拿到**独立**证据，部件必须经过独立前向：按掩码裁区域 → 重新过 backbone。
#   仓库里的 `part_rerank` 正是这条路，但历史上它失败的原因是掩码是碎斑——
#   每个部件的框覆盖全图 98.5%，裁出来还是整图，等于把原图再看一遍。
#
#   现在熵上限可以把掩码压成真部件（重叠 0.009、覆盖 14% 的位置，§21），
#   所以"裁出来重编码"第一次有了意义。两格成对，只差熵正则：
#     all_rr_pa4k8      —— 碎斑掩码 + 重编码（历史上失败的那个配置，全训口径的对照）
#     all_rr_pa4k8_ent  —— 真部件掩码 + 重编码（新）
#   两格都开 part_rerank，训练/测试同一条路径（K+1 次前向，测试由模型内部完成，符合约束）。
#
# 排在本批最后，等 chain_res 的评测跑完再上。
cd /root/autodl-tmp/cub-bird || exit 1
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
LOG=/root/chain_rerank.log
for _ in $(seq 1 900); do grep -q "CHAIN_RES_DONE" /root/chain_res.log 2>/dev/null && break; sleep 60; done
echo "[$(date '+%F %T')] 前置批次结束，启动重编码两格" >> "$LOG"

BASE="--config configs/baseline_config.yaml --set DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=8 TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3 MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1 TRAIN.swa_start_frac=[0.5,0.6,0.7] DATASET.train_on_val=true"
PA8="MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=8 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 TRAIN.pa_div_w=1.0 TRAIN.pa_aux_w=0.5 TRAIN.pa_lr=5.0e-4"
RR="MODEL.part_rerank=true MODEL.pr_parts=4 MODEL.pr_margin=1.3 MODEL.pr_alpha=0.2"

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

launch 0 r50_all_rr_pa4k8     $PA8 $RR
launch 1 r50_all_rr_pa4k8_ent $PA8 $RR MODEL.pa_ent_w=1.0 MODEL.pa_ent_target=3.5 MODEL.pa_ent_ramp=0.3
waitnames "EXP_NAME=(r50_all_rr_pa4k8|r50_all_rr_pa4k8_ent)$"
evalbatch r50_all_rr_pa4k8 r50_all_rr_pa4k8_ent
cleanup    r50_all_rr_pa4k8 r50_all_rr_pa4k8_ent
echo "[$(date '+%F %T')] CHAIN_RR_DONE" >> "$LOG"
