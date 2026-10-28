#!/usr/bin/env bash
# 在 queueRR 跑完 r50_gt_head_hl 之后抢占，改跑 queueSS（part_pad 修正）。
# 理由：queueRR 剩下的 soft/c05/nocons 远程 batch4 正在同时跑（同配置），本地不重复；
# 本地改跑「头框外扩」这个修正，它针对的是刚发现的 cropCE 偏高问题。
set -u
cd /home/wpx/main/work/my-notes/projects/cub-bird || exit 1
LOG=logs/queue_r50_90.log
for _ in $(seq 1 240); do
  if grep -q "START r50_gt_head_soft" "$LOG"; then
    echo "[$(date '+%F %T')] 检测到 r50_gt_head_soft 起头，抢占改跑 queueSS" >> "$LOG"
    pkill -f "EXP_NAME=r50_gt_head_soft" 2>/dev/null
    pkill -f "queue_r50_90_rr.sh" 2>/dev/null
    sleep 5
    bash scripts/queue_r50_90_ss.sh
    exit 0
  fi
  if grep -q "queueRR 结束" "$LOG"; then
    bash scripts/queue_r50_90_ss.sh
    exit 0
  fi
  sleep 20
done
