#!/usr/bin/env bash
set -u
cd /home/wpx/main/work/my-notes/projects/cub-bird || exit 1
LOG=logs/queue_r50_90.log
for _ in $(seq 1 400); do
  if grep -q "DONE r50_attnp_cls_rep" "$LOG"; then
    echo "[$(date '+%F %T')] 替换出口跑完，接跑与 GAP 拼接的出口" >> "$LOG"
    bash scripts/queue_attnp_cat.sh
    exit 0
  fi
  sleep 30
done
