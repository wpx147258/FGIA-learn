#!/usr/bin/env bash
set -u
cd /home/wpx/main/work/my-notes/projects/cub-bird || exit 1
LOG=logs/queue_r50_90.log
for _ in $(seq 1 500); do
  if grep -q "DONE r50_all_attnpcls_s1" "$LOG"; then
    echo "[$(date '+%F %T')] attnpcls_s1 结束，接跑同机同码同种子的复现测试" >> "$LOG"
    bash scripts/queue_pa4js_repro_test.sh
    exit 0
  fi
  sleep 30
done
