#!/usr/bin/env bash
# 诊断队列：等 JS 版跑完后，用 probe_token_stats 量熵/JS/IoU，判断约束是否生效。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/wpx/anaconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com
LOG=logs/queue_r50_90.log
echo "[$(date '+%F %T')] queueNN 等待 JS 版实验结束 ..." >> "$LOG"
until grep -qs "queueLL 结束" "$LOG"; do sleep 30; done
for name in r50_tl_js r50_pa4_js; do
  d=$(ls -dt outputs/${name}/*/ 2>/dev/null | head -1)
  [ -z "$d" ] && continue
  "$PY" scripts/probe_token_stats.py --checkpoint "${d}swa_f0.5.pth" --n 96 --batch-size 8 \
    > "logs/${name}_stats.log" 2>&1 || true
  echo "[$(date '+%F %T')] STATS $name -> logs/${name}_stats.log" >> "$LOG"
done
echo "[$(date '+%F %T')] queueNN 结束" >> "$LOG"
