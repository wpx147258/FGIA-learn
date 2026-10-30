#!/usr/bin/env bash
# 2026-10-30：用户判断——`e75m2_pa4k8` 掉 0.64 不能说明"PA 有害"，
# 因为那时 PA 的掩码到今天为止都不是真部件（外接框=整个输入窗口）。
# 现在有了「紧致 + 边界惩罚 + 反重复」三件套（10 轮验证：框面积 51%→14%，且落在不同位置），
# 必须用**成形的部件**重做这个组合。
#
# 参照（本地同机同 SEED=1）：`e75m2_s1` = 89.58（f0.6）。
# 若这一格明显低于 89.58，才是"PA 真的有害"；若持平/更高，说明之前的负面结论是掩码缺陷造成的。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/queue_r50_90.log
echo "[$(date '+%F %T')] queue_local_e75m2_pacmp 等待 ovl05_s1 结束 ..." >> "$LOG"
until grep -qs "queue_local_overlap 结束" "$LOG"; do sleep 30; done

BASE=(
  --config configs/baseline_config.yaml --set
  DATASET.image_size=448 TRAIN.epochs=75 TRAIN.batch_size=16
  TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05
  TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3
  TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0
  TRAIN.cam_margin2=1.6 TRAIN.cam_consistency2=0.15
  TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01
  TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05
  TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3
  MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1
  "TRAIN.swa_start_frac=[0.6, 0.7, 0.8]" DATASET.train_on_val=true
  MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=8 MODEL.pa_grouped=true
  MODEL.pa_compete=false MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1
  MODEL.pa_compact_w=0.5 MODEL.pa_dup_w=1.0 MODEL.pa_dup_thr=0.9 MODEL.pa_border_w=1.0
  TRAIN.pa_div_w=0.0 TRAIN.pa_aux_w=0.5 TRAIN.pa_lr=5.0e-4
)
name=r50_all_e75m2_pacmp_s1
echo "[$(date '+%F %T')] START $name" >> "$LOG"
"$PY" train.py "${BASE[@]}" SEED=1 EXP_NAME="$name" > "logs/${name}_train.log" 2>&1
d=$(ls -dt "outputs/${name}"/*/ 2>/dev/null | head -1)
echo "[$(date '+%F %T')] DONE $name -> $d" >> "$LOG"
"$PY" scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] queue_local_e75m2_pacmp 结束" >> "$LOG"
