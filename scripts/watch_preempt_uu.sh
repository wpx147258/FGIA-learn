#!/usr/bin/env bash
# queueTT 跑到第二格（cam_wing_pc）时抢占，改跑「头+翅」的两个 seed 重复。
set -u
cd /home/wpx/main/work/my-notes/projects/cub-bird || exit 1
LOG=logs/queue_r50_90.log
for _ in $(seq 1 400); do          # 最多等 ~2.2 小时
  if grep -q "START r50_cam_wing_pc" "$LOG"; then
    echo "[$(date '+%F %T')] 检测到 r50_cam_wing_pc 起头，抢占改跑 queueUU" >> "$LOG"
    pkill -f "EXP_NAME=r50_cam_wing_pc" 2>/dev/null
    pkill -f "queue_r50_90_tt.sh" 2>/dev/null
    sleep 5
    bash scripts/queue_r50_90_uu.sh
    exit 0
  fi
  if grep -q "queueTT 结束" "$LOG"; then
    bash scripts/queue_r50_90_uu.sh
    exit 0
  fi
  sleep 20
done
echo "[$(date '+%F %T')] 等待 queueTT 超时，未启动 queueUU" >> "$LOG"
