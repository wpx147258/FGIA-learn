#!/usr/bin/env bash
set -u
cd /home/wpx/main/work/my-notes/projects/cub-bird || exit 1
LOG=logs/queue_r50_90.log
for _ in $(seq 1 400); do
  if grep -q "DONE r50_pa4_l3_k4b" "$LOG"; then
    echo "[$(date '+%F %T')] l3_k4b 结束，接跑紧熵上限两格" >> "$LOG"
    bash scripts/queue_local_ceil.sh
    exit 0
  fi
  sleep 30
done
