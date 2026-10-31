#!/usr/bin/env bash
# 2026-10-31：**局部池化**——部件描述子只从「承载前 20% 质量」的格子里取。
#
# 依据（今天的框分析）：四项目版掩码按「累计 20% 质量」画框时，框面积 18.5%、
# 框内 GT 覆盖率 **88.1%**（按 80% 规则画则是 95%/54%）——局部结构藏在质量尾部，
# 而现在的池化把整张掩码当权重、等于又摊回整图。
# 这个开关让池化只用那一小块，是"让掩码结构真正进入分类器"最直接的一步。
#
# 配置：**标准 PA（compete=true）** + top_frac 0.2 / 0.1。
# ⚠️ 不用四项目版：它 K_eff=1.05（8 个部件是同一张拷贝），拿它做局部池化没有意义。
# 标准 PA 的 K_eff=6.92（真分化），每个部件平均占 1/8 图；取其前 20% 质量 ≈ 2.5% 的局部区域。
# 参照：同机同种子的 `pa4k8_full_s1` = 89.28、`base_s1` = 89.26。
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
  TRAIN.pa_div_w=1.0 TRAIN.pa_aux_w=0.5 MODEL.pa_js_w=0.0 TRAIN.pa_lr=5.0e-4
)
for f in 0.2 0.1; do
  name=r50_all_pa4k8_stdtop$(echo $f | tr -d '.')
  echo "[$(date '+%F %T')] START $name" >> "$LOG"
  "$PY" train.py "${BASE[@]}" MODEL.pa_top_frac=$f SEED=1 EXP_NAME="$name" > "logs/${name}_train.log" 2>&1
  d=$(ls -dt "outputs/${name}"/*/ 2>/dev/null | head -1)
  echo "[$(date '+%F %T')] DONE $name -> $d" >> "$LOG"
  [ -n "$d" ] && "$PY" scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1
done
echo "[$(date '+%F %T')] queue_local_topfrac 结束" >> "$LOG"
