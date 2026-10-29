#!/usr/bin/env bash
set -u
cd /home/wpx/main/work/my-notes/projects/cub-bird || exit 1
LOG=logs/queue_r50_90.log
for _ in $(seq 1 400); do
  if grep -q "DONE r50_attnp_cat" "$LOG"; then
    echo "[$(date '+%F %T')] attnp_cat 结束，重跑替换出口（修掉 trainer 里 ap.out 的 AttributeError）" >> "$LOG"
    bash scripts/queue_attnp_replace.sh
    exit 0
  fi
  sleep 30
done
