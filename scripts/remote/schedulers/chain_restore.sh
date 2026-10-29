#!/usr/bin/env bash
# 等第一轮（gc / gc_ctrl / clip_gap / clip_ap）全部结束后，再跑三轮补齐
cd /root/autodl-tmp/cub-bird || exit 1
started=0
for _ in $(seq 1 600); do
  n=$(ls -d outputs/r50_gc outputs/r50_gc_ctrl outputs/r50_clip_gap outputs/r50_clip_ap 2>/dev/null | wc -l)
  [ "$n" -ge 4 ] && started=1
  running=$(pgrep -fc "EXP_NAME=r50_gc|EXP_NAME=r50_gc_ctrl|EXP_NAME=r50_clip_gap|EXP_NAME=r50_clip_ap" || true)
  if [ "$started" = "1" ] && [ "${running:-0}" -eq 0 ]; then
    echo "[$(date '+%F %T')] 第一轮结束，开始补齐" >> /root/restore_plan.log
    break
  fi
  sleep 60
done
bash /root/restore_plan.sh >> /root/restore_plan.log 2>&1
