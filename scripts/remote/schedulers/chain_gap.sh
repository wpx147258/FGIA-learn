#!/usr/bin/env bash
# 补齐 QUEUE_NOW §6 的缺口（全训协议）：分三批，每批跑完再下一批。
#
# 2026-10-30 01:55 修订：
#   - G3 空出的第 4 个槽位补上 `r50_all_pa4js_blur_v2`：原来的 `*_blur` 四格因为
#     `pa_mask_blur` 没进前向而全部等于没开（见 9cf2531），需要按修复后的代码重跑一格，
#     否则「碎斑→区域」这条假设至今没有被真正检验过。
#   - `wait` 改成按 EXP_NAME 轮询。
cd /root/autodl-tmp/cub-bird || exit 1
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
LOG=/root/chain_gap.log
for _ in $(seq 1 600); do grep -q "CHAIN_LONG_DONE" /root/chain_long.log 2>/dev/null && break; sleep 60; done
echo "[$(date '+%F %T')] 前置批次结束，开始缺口三批" >> "$LOG"
BASE="--config configs/baseline_config.yaml --set DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16 TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3 MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1 TRAIN.swa_start_frac=[0.5,0.6,0.7] DATASET.train_on_val=true"
PA="MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=4 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 TRAIN.pa_div_w=1.0 TRAIN.pa_aux_w=0.5 TRAIN.pa_lr=5.0e-4"
AT="MODEL.attn_pool=true MODEL.attn_readout=cls MODEL.attn_depth=1 MODEL.attn_dropout=0.0"
PA8="MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=8 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 TRAIN.pa_div_w=1.0 TRAIN.pa_aux_w=0.5 TRAIN.pa_lr=5.0e-4"
launch () { local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY train.py $BASE "$@" EXP_NAME="$name" > "logs/${name}_train.log" 2>&1 &
  echo "[$(date '+%F %T')] launch $name gpu$gpu" >> "$LOG"; }
waitnames () { local pat="$1"
  for _ in $(seq 1 60); do pgrep -f "$pat" >/dev/null 2>&1 && break; sleep 20; done
  while pgrep -f "$pat" >/dev/null 2>&1; do sleep 30; done; }
evalbatch () { for n in "$@"; do d=$(ls -dt outputs/$n/*/ 2>/dev/null|head -1)
  [ -n "$d" ] && { echo "--- eval $n" >> "$LOG"; $PY scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1; }; done; }

echo "=== G1 ===" >> "$LOG"
launch 0 r50_all_t49       $AT MODEL.attn_tokens=49
launch 1 r50_all_attnp_d0  $AT MODEL.attn_depth=0
launch 2 r50_all_pa4k8_cls $PA MODEL.pa_parts=8 MODEL.pa_readout=cls
# τ 扫描（tau005/tau02）按 2026-10-30 的诊断撤掉：掩码本来就是近 one-hot，
# τ 动不了它；换成诊断直接指向的「熵上限」两格（部件越训越平是实测到的失效模式）。
launch 3 r50_all_pa4k8_ent1 $PA8 MODEL.pa_ent_w=1.0 MODEL.pa_ent_target=3.5 MODEL.pa_ent_ramp=0.3
waitnames "EXP_NAME=(r50_all_t49|r50_all_attnp_d0|r50_all_pa4k8_cls|r50_all_pa4k8_ent1)$"
evalbatch r50_all_t49 r50_all_attnp_d0 r50_all_pa4k8_cls r50_all_pa4k8_ent1

echo "=== G2 ===" >> "$LOG"
launch 0 r50_all_pa4k8_ent3 $PA8 MODEL.pa_ent_w=3.0 MODEL.pa_ent_target=2.5 MODEL.pa_ent_ramp=0.3
launch 1 r50_all_aux1      $PA TRAIN.pa_aux_w=1.0
launch 2 r50_all_pos       $AT MODEL.attn_pos=true
launch 3 r50_all_dim512    $AT MODEL.attn_dim=512
waitnames "EXP_NAME=(r50_all_pa4k8_ent3|r50_all_aux1|r50_all_pos|r50_all_dim512)$"
evalbatch r50_all_pa4k8_ent3 r50_all_aux1 r50_all_pos r50_all_dim512

echo "=== G3 ===" >> "$LOG"
launch 0 r50_all_heads8    $AT MODEL.attn_heads=8
launch 1 r50_all_div03     $PA TRAIN.pa_div_w=0.3
launch 2 r50_all_div30     $PA TRAIN.pa_div_w=3.0
launch 3 r50_all_pa4js_blur_v2 $PA MODEL.pa_js_w=0.5 MODEL.pa_js_target=0.35 MODEL.pa_mask_blur=true
waitnames "EXP_NAME=(r50_all_heads8|r50_all_div03|r50_all_div30|r50_all_pa4js_blur_v2)$"
evalbatch r50_all_heads8 r50_all_div03 r50_all_div30 r50_all_pa4js_blur_v2
echo "[$(date '+%F %T')] CHAIN_GAP_DONE" >> "$LOG"
