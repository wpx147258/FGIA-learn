#!/usr/bin/env bash
set -u
cd /home/wpx/main/work/my-notes/projects/cub-bird || exit 1
LOG=logs/queue_r50_90.log
for _ in $(seq 1 400); do
  if grep -q "queue_attn_d2 结束" "$LOG"; then
    echo "[$(date '+%F %T')] depth=2 两格结束，接跑 layer3 K=4" >> "$LOG"
    bash scripts/queue_l3_k4.sh
    exit 0
  fi
  sleep 20
done
