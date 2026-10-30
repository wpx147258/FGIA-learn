#!/usr/bin/env bash
# 2026-10-30 10:00：PartAttention 的**正则拆解**批（全训，SEED=42，50 轮）。
#
# 背景（2026-10-30 审计）：本地 26 个 PA run **没有一个**关掉过模块正则——
#   (div, aux, js) 的取值只有 (1.0,0.5,0)×22、(1.0,0.25,0)×3、(1.0,0.5,0.5)×3、(2.0,0.5,0)×2、(0.1,0.5,0)×1。
#   也就是说所有 PA 结论都是「模块 + 正则」打包的结果，从没测过"裸模块"。
#   G3 里 div 权重 0.3→3.0 只动 0.14，正是因为从未测过 0 那一端。
#
# 四格把 pa4k8（89.75）拆开：
#   all_pa4k8_noreg  div=0 aux=0 js=0  —— 裸模块，回答"部件注意力本身值多少"
#   all_pa4k8_div0   div=0 aux=0.5     —— 只去掉散度项
#   all_pa4k8_aux0   div=1.0 aux=0     —— 只去掉逐部件 CE
#   all_pa4_noreg    K=4 全关           —— 无正则下 K 还有没有差别
#
# 排在 chain_stack（e75m2 叠加批）之后、true_repro 之前。
cd /root/autodl-tmp/cub-bird || exit 1
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
LOG=/root/chain_noreg.log
for _ in $(seq 1 900); do grep -q "CHAIN_STACK_DONE" /root/chain_stack.log 2>/dev/null && break; sleep 60; done
echo "[$(date '+%F %T')] 叠加批结束，启动 PA 正则拆解四格" >> "$LOG"

BASE="--config configs/baseline_config.yaml --set DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16 TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3 MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1 TRAIN.swa_start_frac=[0.5,0.6,0.7] DATASET.train_on_val=true"
PA="MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 TRAIN.pa_lr=5.0e-4"

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

launch 0 r50_all_pa4k8_noreg $PA MODEL.pa_parts=8 TRAIN.pa_div_w=0.0 TRAIN.pa_aux_w=0.0 MODEL.pa_js_w=0.0
launch 1 r50_all_pa4k8_div0  $PA MODEL.pa_parts=8 TRAIN.pa_div_w=0.0 TRAIN.pa_aux_w=0.5
launch 2 r50_all_pa4k8_aux0  $PA MODEL.pa_parts=8 TRAIN.pa_div_w=1.0 TRAIN.pa_aux_w=0.0
launch 3 r50_all_pa4_noreg   $PA MODEL.pa_parts=4 TRAIN.pa_div_w=0.0 TRAIN.pa_aux_w=0.0 MODEL.pa_js_w=0.0

waitnames "EXP_NAME=(r50_all_pa4k8_noreg|r50_all_pa4k8_div0|r50_all_pa4k8_aux0|r50_all_pa4_noreg)$"
echo "[$(date '+%F %T')] PA 正则拆解四格训练结束，评测" >> "$LOG"
for n in r50_all_pa4k8_noreg r50_all_pa4k8_div0 r50_all_pa4k8_aux0 r50_all_pa4_noreg; do
  d=$(ls -dt outputs/$n/*/ 2>/dev/null|head -1)
  [ -n "$d" ] && { echo "--- eval $n" >> "$LOG"; $PY scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1; }
done
cleanup r50_all_pa4k8_noreg r50_all_pa4k8_div0 r50_all_pa4k8_aux0 r50_all_pa4_noreg
echo "[$(date '+%F %T')] CHAIN_NOREG_DONE" >> "$LOG"
