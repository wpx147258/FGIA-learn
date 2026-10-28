#!/usr/bin/env bash
# 等 swsl_448_camcrop02_b01（backbone_lr_scale 固定 0.1）跑完后，
# 立刻用「渐进解冻」配置起下一轮：
#   backbone_lr_scale 0.1 -> 最后 0.3，在总轮数的 10%~60% 之间线性过渡
#
# 用法（脱离当前 shell）：
#   setsid nohup bash scripts/queue_swsl_prog.sh < /dev/null &
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/home/wpx/anaconda3/bin/python
CUR='EXP_NAME=swsl_448_camcrop02_b01'
LOG=outputs/_logs/queue_swsl_prog.log

echo "[$(date '+%F %T')] 等待 b01 结束 ..." >> "$LOG"
while pgrep -f "$CUR" >/dev/null 2>&1; do
    sleep 15
done
echo "[$(date '+%F %T')] GPU 空闲，启动 prog run" >> "$LOG"

setsid nohup systemd-inhibit --what=sleep --mode=block "$PY" train.py \
    --config configs/baseline_config.yaml --set \
    DATASET.image_size=448 \
    TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.epochs=50 TRAIN.batch_size=16 \
    TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.3 TRAIN.cam_margin=1.3 \
    TRAIN.backbone_lr_scale=0.1 TRAIN.backbone_lr_scale_final=0.3 \
    "TRAIN.backbone_lr_ramp=[0.1, 0.6]" \
    MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k \
    EXP_NAME=swsl_448_camcrop02_prog >> "$LOG" 2>&1 < /dev/null &

echo "[$(date '+%F %T')] 已启动 pid=$! （日志 $LOG）" >> "$LOG"
