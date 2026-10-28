#!/usr/bin/env bash
# 等 swsl_448_layers_nocutmix 跑完后，起「min_lr 2e-6」那一轮。
#
# 相对 swsl_448_layers（test 87.50）只改 min_lr：1e-6 -> 2e-6。
# 注意 yaml 的写法：2e-6 会被解析成**字符串**（YAML 1.1 要求科学计数法带小数点），
# 必须写 2.0e-6，否则 min_lr / lr 会报 TypeError。
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/home/wpx/anaconda3/bin/python
CUR='EXP_NAME=swsl_448_layers_nocutmix'
LOG=outputs/_logs/queue_minlr.log

echo "[$(date '+%F %T')] 等待 nocutmix 结束 ..." >> "$LOG"
while pgrep -f "$CUR" >/dev/null 2>&1; do
    sleep 15
done
echo "[$(date '+%F %T')] GPU 空闲，启动 min_lr=2e-6" >> "$LOG"

setsid nohup systemd-inhibit --what=sleep --mode=block "$PY" train.py \
    --config configs/baseline_config.yaml --set \
    DATASET.image_size=448 \
    TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.epochs=50 TRAIN.batch_size=16 \
    TRAIN.min_lr=2.0e-6 \
    TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.3 TRAIN.cam_margin=1.3 \
    TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 \
    TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 \
    TRAIN.backbone_lr_layers.layer4=0.2 \
    "TRAIN.swa_start_frac=[0.5, 0.6, 0.7]" \
    MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k \
    EXP_NAME=swsl_448_layers_mlr2e6 >> "$LOG" 2>&1 < /dev/null &

echo "[$(date '+%F %T')] 已启动 pid=$!" >> "$LOG"
