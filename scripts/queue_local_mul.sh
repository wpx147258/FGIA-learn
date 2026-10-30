#!/usr/bin/env bash
# 2026-10-31：**乘性出口**——目前唯一还没被证伪的、能让掩码影响决策的机制。
#
# 依据（今天下午的因果干预，n=5794 全 test）：
#   zero-res(去掉分支贡献) 89.07 / baseline 89.35 / uniform-mask 89.35(翻转 0 张) /
#   rolled-mask 89.35(翻转 0 张)
#   ⇒ 在**加性出口**下，掩码内容对预测的因果贡献是 0。
#   整条分支那 +0.28 全部来自"多了一条参数化加性通路"。
#
# 所以唯一还可能让掩码有用的改动是**换出口**：乘性 gain `output * (1 + γ·Σ_k c_k m_k)`
# 让主分类 loss 第一次"看见"掩码的空间位置。
# 之前的 pa4k8_mul 用过 gate=-2 但掩码是"框覆盖 95% 的碎斑"，等于没做空间重加权；
# 这一格用**紧致 + 边界**（框 29%、不贴边）的掩码，gate=-4。
#
# 关键诊断：日志里新增 `pa_mul_gamma`。若 γ 训练中一直贴在初值 0.018 附近，
# 说明分类目标根本不需要空间重加权——这是"该收尾"的决定性读数。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/queue_r50_90.log
BASE=(
  --config configs/baseline_config.yaml --set
  DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16
  TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05
  TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3
  TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0
  TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01
  TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05
  TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3
  MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1
  "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]" DATASET.train_on_val=true
  MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=8 MODEL.pa_grouped=true
  MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1
  MODEL.pa_compact_w=0.5 MODEL.pa_border_w=1.0
  MODEL.pa_mul=true MODEL.pa_mul_gate=-4.0 MODEL.pa_mul_center=true
  TRAIN.pa_div_w=1.0 TRAIN.pa_aux_w=0.5 TRAIN.pa_lr=5.0e-4
)
name=r50_all_pa4k8_cmpb_mul
echo "[$(date '+%F %T')] START $name" >> "$LOG"
"$PY" train.py "${BASE[@]}" SEED=1 EXP_NAME="$name" > "logs/${name}_train.log" 2>&1
d=$(ls -dt "outputs/${name}"/*/ 2>/dev/null | head -1)
echo "[$(date '+%F %T')] DONE $name -> $d" >> "$LOG"
"$PY" scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] queue_local_mul 结束" >> "$LOG"
