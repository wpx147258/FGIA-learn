#!/usr/bin/env bash
# 2026-10-30 02:45：把前面几次重排里**被换掉/删掉**的实验原样补回来，一格都不丢。
#
# 丢掉的五格（每一次都是我自己重排时换下的，不是跑失败）：
#   r50_all_tau005       τ=0.05        —— 原 chain_gap G1，被熵上限格替换
#   r50_all_tau02        τ=0.2         —— 原 chain_gap G2，被熵上限格替换
#   r50_all_sigk8        K=8 + 无 compete —— 原 chain_pa 第 2 格
#   r50_all_sigk8_blur   K=8 + 无 compete + 平滑 —— 原 chain_pa 第 3 格
#   r50_all_pa4js_l3     K=4 + JS + layer3 —— 原 chain_pa 第 4 格
# 另加 r50_all_k2（本地排过、后来被基线复现顶掉的 K=2）。
#
# 排在整条链的最后（TRUE_REPRO_DONE 之后），完全不动已有队列的顺序。
cd /root/autodl-tmp/cub-bird || exit 1
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
LOG=/root/chain_restore.log
# 先让位给新实验批（chain_res：分辨率 + CAM 强度），补回的老实验排在它们之后
for _ in $(seq 1 900); do grep -q "CHAIN_RES_DONE" /root/chain_res.log 2>/dev/null && break; sleep 60; done
echo "[$(date '+%F %T')] 前置链路全部结束，开始补回被换下的实验" >> "$LOG"

BASE="--config configs/baseline_config.yaml --set DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16 TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3 MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1 TRAIN.swa_start_frac=[0.5,0.6,0.7] DATASET.train_on_val=true"
PA="MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=4 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 TRAIN.pa_div_w=1.0 TRAIN.pa_aux_w=0.5 TRAIN.pa_lr=5.0e-4"
SIG="MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=8 MODEL.pa_grouped=true MODEL.pa_compete=false MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 TRAIN.pa_div_w=1.0 TRAIN.pa_aux_w=0.5 TRAIN.pa_lr=5.0e-4"
AT="MODEL.attn_pool=true MODEL.attn_readout=cls MODEL.attn_depth=1 MODEL.attn_dropout=0.0"

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

echo "=== R1：τ 两格 + 无 compete 的 K=8 两格 ===" >> "$LOG"
launch 0 r50_all_tau005     $PA MODEL.pa_tau=0.05
launch 1 r50_all_tau02      $PA MODEL.pa_tau=0.2
launch 2 r50_all_sigk8      $SIG
launch 3 r50_all_sigk8_blur $SIG MODEL.pa_mask_blur=true
waitnames "EXP_NAME=(r50_all_tau005|r50_all_tau02|r50_all_sigk8|r50_all_sigk8_blur)$"
evalbatch r50_all_tau005 r50_all_tau02 r50_all_sigk8 r50_all_sigk8_blur
cleanup    r50_all_tau005 r50_all_tau02 r50_all_sigk8 r50_all_sigk8_blur

echo "=== R2：layer3 版 + K=2 + AttnPool mean 读出 ===" >> "$LOG"
launch 0 r50_all_pa4js_l3   $PA MODEL.pa_from=layer3 MODEL.pa_js_w=0.5 MODEL.pa_js_target=0.35
# K=2 原来是本地脚本里的 compete=false；现在已知 compete=false 会把 K 张掩码塌成同一张
# （余弦 0.9998），所以按 compete=true 跑，与 pa4k8/pa4 同族。
launch 1 r50_all_k2         MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=2 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 TRAIN.pa_div_w=1.0 TRAIN.pa_aux_w=0.5 TRAIN.pa_lr=5.0e-4
# 这一格不是"补回"，是补一个从没跑过的缺口：AttnPool 在全训口径只有 CLS 读出，mean 一次没跑过。
launch 2 r50_all_attnp_mean $AT MODEL.attn_readout=mean
waitnames "EXP_NAME=(r50_all_pa4js_l3|r50_all_k2|r50_all_attnp_mean)$"
evalbatch r50_all_pa4js_l3 r50_all_k2 r50_all_attnp_mean
cleanup    r50_all_pa4js_l3 r50_all_k2 r50_all_attnp_mean
echo "[$(date '+%F %T')] CHAIN_RESTORE_DONE" >> "$LOG"
