#!/usr/bin/env bash
# 等 pa4k8_noreg_s7 训完，自动出两张图：外接框 + 掩码热力图。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
LOG=logs/queue_r50_90.log
while pgrep -f "EXP_NAME=r50_all_pa4k8_noreg_s7$" >/dev/null 2>&1; do sleep 30; done
D=$(ls -dt outputs/r50_all_pa4k8_noreg_s7/*/ 2>/dev/null | head -1)
[ -z "$D" ] && { echo "[$(date '+%F %T')] auto_vis: 找不到输出目录" >> "$LOG"; exit 1; }
CK="${D}swa_f0.5.pth"; [ -f "$CK" ] || CK="${D}last.pth"
echo "[$(date '+%F %T')] auto_vis noreg: 用 $CK" >> "$LOG"
"$PY" scripts/vis_part_boxes.py --checkpoint "$CK" --n 24 --stride --out figs/part_boxes_noreg_x24.png >> "$LOG" 2>&1
"$PY" scripts/vis_part_masks.py --checkpoint "$CK" --n 8 --out figs/part_masks_noreg.png >> "$LOG" 2>&1
echo "[$(date '+%F %T')] auto_vis noreg 完成" >> "$LOG"
