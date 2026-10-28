#!/usr/bin/env bash
# queueSS 两格跑完后接 queueTT（额外部位视图 + 一致性）。
set -u
cd /home/wpx/main/work/my-notes/projects/cub-bird || exit 1
LOG=logs/queue_r50_90.log
for _ in $(seq 1 240); do
  if grep -q "queueSS 结束" "$LOG"; then
    echo "[$(date '+%F %T')] 检测到 queueSS 结束，启动 queueTT" >> "$LOG"
    bash scripts/queue_r50_90_tt.sh
    exit 0
  fi
  sleep 20
done
echo "[$(date '+%F %T')] 等待 queueSS 超时，未启动 queueTT" >> "$LOG"
