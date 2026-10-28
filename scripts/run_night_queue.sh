#!/bin/bash
# 夜间实验队列：串行跑 TODO.md 里的实验，每个跑完自动测 test 并记到 outputs/_queue.log。
# 单个实验失败不阻塞后续（记录 rc，继续下一个）。
set -u
cd /home/wpx/main/work/my-notes/projects/cub-bird
PY=/home/wpx/anaconda3/bin/python
BASE="DATASET.image_size=448 TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.label_smoothing=0.1"
LOG=outputs/_queue.log

run_one () {
  local name="$1"; shift
  local sets="$*"
  echo "[$(date '+%F %T')] QUEUE START $name" >> "$LOG"
  PYTHONUNBUFFERED=1 systemd-inhibit --what=sleep --mode=block \
    "$PY" train.py --config configs/baseline_config.yaml \
    --set $BASE $sets EXP_NAME="$name" > "outputs/_logs/$name.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] QUEUE END   $name rc=$rc" >> "$LOG"

  local d
  d=$(ls -d "outputs/$name"/*/ 2>/dev/null | tail -1)
  if [ -n "$d" ]; then
    for c in best last swa; do
      if [ -f "$d/$c.pth" ]; then
        local r
        r=$("$PY" test.py --checkpoint "$d/$c.pth" 2>&1 | grep '^\[test\]')
        echo "[$(date '+%F %T')] TEST $name/$c  $r" >> "$LOG"
      fi
    done
  fi
}

# 1. cam_crop=0.1 —— 补完权重曲线（0 / 0.2 / 0.5 已测）
run_one a1_448_camcrop_w01 \
  "TRAIN.epochs=50 TRAIN.cam_crop=0.1 TRAIN.cam_alpha_crop=0.3 TRAIN.cam_margin=1.3"

# 2. cam_crop=0.2 + 100 epoch
run_one a1_448_camcrop_w02_e100 \
  "TRAIN.epochs=100 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.3 TRAIN.cam_margin=1.3"

# 3a. crop + 软擦除，权重 0.2（擦除后 81.6%，难度比主分支低 5 点）
run_one a1_448_camcrop_softdrop_w02 \
  "TRAIN.epochs=50 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.3 TRAIN.cam_margin=1.3 \
   TRAIN.cam_drop=0.2 TRAIN.cam_alpha_drop=0.3 TRAIN.cam_drop_strength=0.7"

# 3b. 同上，权重减半
run_one a1_448_camcrop_softdrop_w01 \
  "TRAIN.epochs=50 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.3 TRAIN.cam_margin=1.3 \
   TRAIN.cam_drop=0.1 TRAIN.cam_alpha_drop=0.3 TRAIN.cam_drop_strength=0.7"

echo "[$(date '+%F %T')] QUEUE ALL DONE" >> "$LOG"
