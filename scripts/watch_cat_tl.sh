#!/usr/bin/env bash
set -u
cd /home/wpx/main/work/my-notes/projects/cub-bird || exit 1
LOG=logs/queue_r50_90.log
for _ in $(seq 1 400); do
  if grep -q "DONE r50_attnp_cat" "$LOG"; then
    echo "[$(date '+%F %T')] AttnPool 拼接跑完，接跑 TokenLearner 拼接两格" >> "$LOG"
    bash scripts/queue_tl_cat_gap.sh
    exit 0
  fi
  sleep 30
done
