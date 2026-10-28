#!/usr/bin/env bash
# 在 queueQQ 跑完 r50_tl_rep_floor_aux 之后抢占，改跑 GT 关键点裁剪（queueRR）。
#
# 理由：queueQQ 剩下的 js_only / floor_only 是 mean 读出的消融，而我们的证据已经说明
# 「mean 读出下分化不产生增益」；GT 裁剪是唯一有正证据的合法通道，优先级更高。
set -u
cd /home/wpx/main/work/my-notes/projects/cub-bird || exit 1
LOG=logs/queue_r50_90.log
for _ in $(seq 1 240); do
  if grep -q "START r50_tl_rep_js_only" "$LOG"; then
    echo "[$(date '+%F %T')] 检测到 r50_tl_rep_js_only 起头，抢占改跑 queueRR" >> "$LOG"
    pkill -f "EXP_NAME=r50_tl_rep_js_only" 2>/dev/null
    pkill -f "queue_r50_90_qq.sh" 2>/dev/null
    sleep 5
    bash scripts/queue_r50_90_rr.sh
    exit 0
  fi
  if grep -q "queueQQ 结束" "$LOG"; then
    bash scripts/queue_r50_90_rr.sh
    exit 0
  fi
  sleep 20
done
echo "[$(date '+%F %T')] 抢占超时，未启动 queueRR" >> "$LOG"
