#!/usr/bin/env bash
set -u
cd /home/wpx/main/work/my-notes/projects/cub-bird || exit 1
LOG=logs/queue_r50_90.log
for _ in $(seq 1 400); do
  if grep -q "DONE r50_attnp_gate" "$LOG"; then
    echo "[$(date '+%F %T')] attnp_gate 结束，接跑替换 GAP 的 AttnPool（89.23 配置）" >> "$LOG"
    bash scripts/queue_attnp_replace.sh
    exit 0
  fi
  sleep 30
done
