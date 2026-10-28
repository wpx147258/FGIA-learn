#!/bin/bash
# 无 CutMix 重跑队列。
#
# 背景：CutMix 有一半的 batch 贴入面积 > 50%（β=1.0 对称分布），对细粒度是毁灭性的
# ——鸟只占画面 30% 左右，遮一半以上等于把鸟抹掉。而且按面积算的标签和语义质量不符。
# 于是把之前被判"没用"的几个方向，全部放到 cutmix=0 的新基线上重跑一遍，
# 看它们是不是被 CutMix 拖累了。
#
# 所有实验固定：448 / mixup=0.2 / cutmix=0 / 50ep / cam_crop=0.2(α0.3,m1.3) 作为已知有效项。
set -u
cd /home/wpx/main/work/my-notes/projects/cub-bird
PY=/home/wpx/anaconda3/bin/python
BASE="DATASET.image_size=448 TRAIN.mixup=0.2 TRAIN.cutmix=0.0 TRAIN.epochs=50"
LOG=outputs/_queue.log

# 等 mixup_only（p=1.0）跑完，它是这一批的共同基线
while pgrep -f "EXP_NAME=a1_448_camcrop_w02_mixup_only" > /dev/null; do sleep 30; done
echo "[$(date '+%F %T')] NO-CUTMIX QUEUE START" >> "$LOG"

run_one () {
  local name="$1"; shift
  local sets="$*"
  echo "[$(date '+%F %T')] QUEUE START $name" >> "$LOG"
  PYTHONUNBUFFERED=1 systemd-inhibit --what=sleep --mode=block \
    "$PY" train.py --config configs/baseline_config.yaml \
    --set $BASE $sets EXP_NAME="$name" > "outputs/_logs/$name.log" 2>&1
  echo "[$(date '+%F %T')] QUEUE END   $name rc=$?" >> "$LOG"

  local d; d=$(ls -d "outputs/$name"/*/ 2>/dev/null | tail -1)
  if [ -n "$d" ]; then
    for c in best last swa; do
      if [ -f "$d/$c.pth" ]; then
        echo "[$(date '+%F %T')] TEST $name/$c  $($PY test.py --checkpoint "$d/$c.pth" 2>&1 | grep '^\[test\]')" >> "$LOG"
      fi
    done
  fi
}

# 1) 混合比例 0.7 + 干净样本的轻度平滑（ls 只在未混合的 30% 上生效）
run_one a1_448_camcrop_w02_p07_ls005 \
  "TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 \
   TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.3 TRAIN.cam_margin=1.3"

# 2) 四块重排（原 −0.17，最接近成功）
run_one a1_448_camcrop_quad_nocutmix \
  "TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.3 TRAIN.cam_margin=1.3 TRAIN.cam_quad=0.2"

# 3) 软擦除 w=0.1（原 −0.60）
run_one a1_448_camcrop_softdrop_w01_nocutmix \
  "TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.3 TRAIN.cam_margin=1.3 \
   TRAIN.cam_drop=0.1 TRAIN.cam_alpha_drop=0.3 TRAIN.cam_drop_strength=0.7"

# 4) 可学习注意力模块（原 −0.9）
run_one a1_448_attn_nocutmix \
  "TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.3 TRAIN.cam_margin=1.3 \
   TRAIN.attn_weight=0.5 TRAIN.backbone_lr_scale=0.3 MODEL.num_parts=32"

# 5) cam_crop 权重的另一侧（原 0.3 是 −0.67）
run_one a1_448_camcrop_w03_nocutmix \
  "TRAIN.cam_crop=0.3 TRAIN.cam_alpha_crop=0.3 TRAIN.cam_margin=1.3"

echo "[$(date '+%F %T')] NO-CUTMIX QUEUE DONE" >> "$LOG"
