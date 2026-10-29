#!/usr/bin/env bash
set -u
cd /home/wpx/main/work/my-notes/projects/cub-bird || exit 1
LOG=logs/queue_r50_90.log
for _ in $(seq 1 400); do
  if grep -q "DONE r50_attnp_nodrop_s1" "$LOG"; then
    echo "[$(date '+%F %T')] nodrop_s1 结束，接跑 AttnPool 门控读出" >> "$LOG"
    bash scripts/queue_attnp_gate.sh
    exit 0
  fi
  sleep 30
done
