#!/usr/bin/env bash
# 本地 l3_k8 与远程 pa4_l3_k8 重复，跑完 l3_k4 后抢占，改跑 depth=2 的两格。
set -u
cd /home/wpx/main/work/my-notes/projects/cub-bird || exit 1
LOG=logs/queue_r50_90.log
for _ in $(seq 1 400); do
  if grep -q "START r50_pa4_l3_k8" "$LOG"; then
    echo "[$(date '+%F %T')] 检测到 l3_k8 起头（与远程重复），抢占改跑 depth=2" >> "$LOG"
    pkill -f "EXP_NAME=r50_pa4_l3_k8" 2>/dev/null
    pkill -f "queue_pa4_l3.sh" 2>/dev/null
    sleep 5
    bash scripts/queue_attn_d2.sh
    exit 0
  fi
  if grep -q "queue_pa4_l3 结束" "$LOG"; then
    bash scripts/queue_attn_d2.sh; exit 0
  fi
  sleep 20
done
