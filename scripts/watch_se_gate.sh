#!/usr/bin/env bash
set -u
cd /home/wpx/main/work/my-notes/projects/cub-bird || exit 1
LOG=logs/queue_r50_90.log
for _ in $(seq 1 400); do
  if grep -q "DONE r50_se_gate_res" "$LOG"; then
    echo "[$(date '+%F %T')] se_gate 结束，接跑 attnp_nodrop 的 seed 复现" >> "$LOG"
    bash scripts/queue_attnp_nodrop_s1.sh
    exit 0
  fi
  sleep 30
done
