#!/usr/bin/env bash
# 排队：等当前 cbp（50 轮）跑完后，立刻按新配置起下一轮。
#
# 新配置相对当前 run 只改三处（其余逐项相同）：
#   TRAIN.epochs            50  -> 100
#   TRAIN.backbone_lr_scale 1.0 -> 0.5
#   MODEL.dropout           0.0 -> 0.05
#
# 用法（脱离当前 shell，会话结束也不会停）：
#   setsid nohup bash scripts/queue_cbp_e100.sh < /dev/null &
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/home/wpx/anaconda3/bin/python
# 只匹配当前这一轮；新 run 的名字是 a1_448_cbp_e100_b05d05，不会被这里匹配到
CUR='EXP_NAME=a1_448_cbp_camcrop02( |$)'
LOG=outputs/_logs/queue_cbp_e100.log
mkdir -p outputs/_logs

echo "[$(date '+%F %T')] 等待当前训练结束 ..." >> "$LOG"
while pgrep -f "$CUR" >/dev/null 2>&1; do
    sleep 20
done
echo "[$(date '+%F %T')] GPU 空闲，启动 e100 run" >> "$LOG"

# systemd-inhibit 挡住系统休眠：训练跑 80 分钟，中间如果机器睡了就白跑。
setsid nohup systemd-inhibit --what=sleep --mode=block "$PY" train.py \
    --config configs/baseline_config.yaml --set \
    DATASET.image_size=448 \
    TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.epochs=100 \
    TRAIN.batch_size=16 TRAIN.backbone_lr_scale=0.5 \
    MODEL.dropout=0.05 MODEL.head=cbp \
    TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.3 TRAIN.cam_margin=1.3 \
    EXP_NAME=a1_448_cbp_e100_b05d05 \
    >> "$LOG" 2>&1 < /dev/null &

echo "[$(date '+%F %T')] 已启动 pid=$! （日志 $LOG）" >> "$LOG"
