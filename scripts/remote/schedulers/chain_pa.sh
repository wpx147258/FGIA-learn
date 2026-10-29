#!/usr/bin/env bash
# PA 调优批（全训协议）：等 CLS 三格结束 → 跑四格 → 评测。
#
# 为什么这批排在长训练前面：
#   1. `r50_all_parand` 是**决定性对照**——固定随机掩码（不学习）替代学到的掩码。
#      若它 ≈ all_pa4js(89.13)，说明空间结构根本没被学到，后续所有 PA 调参都不必要，
#      可以直接砍掉 chain_gap 里的 tau/aux/div 五格，省 3 个 GPU 小时。
#      若它明显更差，说明学到的东西有用，PA 这条线值得继续调。
#   2. `sigk8` / `sigk8_blur` 是 K=8 上的一对（只差 blur），与已有的
#      all_pa4k8_blur=89.59 组成 compete true/false 的完整对照。
#   3. `pa4js_l3` 把检测器从 14×14 的 layer4 挪到 28×28 的 layer3：
#      探针显示 layer4 特征空间同质、layer3 还有定位能力，这是"掩码不成片"的根因假设之一。
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
PA="MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=4 MODEL.pa_grouped=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 TRAIN.pa_div_w=1.0 TRAIN.pa_aux_w=0.5 TRAIN.pa_lr=5.0e-4 MODEL.pa_compete=false"
launch () { local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY train.py $COMMON "$@" EXP_NAME="$name" > "logs/${name}_train.log" 2>&1 &
  echo "[$(date '+%F %T')] launch $name gpu$gpu" >> "$LOG"; }
waitnames () { local pat="$1"
  for _ in $(seq 1 60); do pgrep -f "$pat" >/dev/null 2>&1 && break; sleep 20; done
  while pgrep -f "$pat" >/dev/null 2>&1; do sleep 30; done; }

# 1) 随机掩码对照：与 all_pa4js 逐字段相同，只把学到的掩码换成固定随机掩码
launch 0 r50_all_parand     $PA MODEL.pa_compete=true MODEL.pa_js_w=0.5 MODEL.pa_js_target=0.35 MODEL.pa_random_masks=true
# 2) K=8 + compete=false（与 all_pa4k8_blur 配对，隔离 compete；与 3) 配对，隔离 blur）
launch 1 r50_all_sigk8      $PA MODEL.pa_parts=8
# 3) K=8 + compete=false + 掩码平滑
launch 2 r50_all_sigk8_blur $PA MODEL.pa_parts=8 MODEL.pa_mask_blur=true
# 4) 检测器放到 layer3（28×28），与 all_pa4js 单变量对照
launch 3 r50_all_pa4js_l3   $PA MODEL.pa_from=layer3 MODEL.pa_compete=true MODEL.pa_js_w=0.5 MODEL.pa_js_target=0.35

waitnames "EXP_NAME=(r50_all_parand|r50_all_sigk8|r50_all_sigk8_blur|r50_all_pa4js_l3)$"
echo "[$(date '+%F %T')] PA 四格训练结束，评测" >> "$LOG"
for n in r50_all_parand r50_all_sigk8 r50_all_sigk8_blur r50_all_pa4js_l3; do
  d=$(ls -dt outputs/$n/*/ 2>/dev/null|head -1)
  [ -n "$d" ] && { echo "--- eval $n" >> "$LOG"; $PY scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1; }
done
echo "[$(date '+%F %T')] CHAIN_PA_DONE" >> "$LOG"
