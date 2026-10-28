#!/usr/bin/env bash
# 在 queuePP 跑到 r50_tl_cls_d2 之前把它停掉，改跑 TL 优先的 queueQQ。
#
# 为什么要抢：queuePP 剩下的 tl_cls_d2 / attnp_cls_d2 是对已经出结论的两条线
# （TL+自注意力 −0.88、AttnPool CLS −0.07）的确认性实验，价值低于 TL 正则消融。
# 这里等 tl_cls_d1 的训练与评估都写完之后、d2 刚起头的时刻切走（最多损失 1 分钟训练）。
set -u
cd /home/wpx/main/work/my-notes/projects/cub-bird || exit 1
LOG=logs/queue_r50_90.log
for _ in $(seq 1 240); do            # 最多等 80 分钟
  if grep -q "START r50_tl_cls_d2" "$LOG"; then
    echo "[$(date '+%F %T')] 检测到 r50_tl_cls_d2 起头，抢占" >> "$LOG"
    pkill -f "EXP_NAME=r50_tl_cls_d2" 2>/dev/null
    pkill -f "queue_r50_90_pp.sh" 2>/dev/null
    sleep 5
    # 原有「等 queuePP 结束」的 watcher 不会再触发，清掉它
    pkill -f "queuePP 结束" 2>/dev/null
    bash scripts/queue_r50_90_qq.sh
    exit 0
  fi
  # queuePP 万一自己正常跑完，也直接接上
  if grep -q "queuePP 结束" "$LOG"; then
    bash scripts/queue_r50_90_qq.sh
    exit 0
  fi
  sleep 20
done
echo "[$(date '+%F %T')] 抢占超时，未启动 queueQQ" >> "$LOG"
