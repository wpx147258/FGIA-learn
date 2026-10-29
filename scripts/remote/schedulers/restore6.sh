#!/usr/bin/env bash
# 分批判执行：每批 4 个（不足 4 个就按实际数），一批全部结束再开下一批。
cd /root/autodl-tmp/cub-bird || exit 1
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
COMMON="--config configs/baseline_config.yaml --set DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16 TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3 MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1 TRAIN.swa_start_frac=[0.5,0.6,0.7]"
launch () { local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY train.py $COMMON "$@" EXP_NAME="$name" > "logs/${name}_train.log" 2>&1 &
  echo "  launch $name on gpu$gpu pid $!"; }
wait_batch () { local pat="$1"
  for _ in $(seq 1 90); do pgrep -f "$pat" >/dev/null 2>&1 && break; sleep 20; done
  while pgrep -f "$pat" >/dev/null 2>&1; do sleep 30; done; }
wait_free4 () { for _ in $(seq 1 240); do
    n=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1<2000' | wc -l)
    [ "$n" -ge 4 ] && sleep 30 && return 0; sleep 30; done; }

echo "=== BATCH 1 ==="; wait_free4
launch 0 r50_attnp_nodrop MODEL.attn_pool=true MODEL.attn_readout=cls MODEL.attn_depth=1 MODEL.attn_dropout=0.0
launch 1 r50_tl_nodrop    MODEL.tokenlearner=true MODEL.tl_readout=cls MODEL.tl_attn_depth=1 MODEL.tl_dropout=0.0
launch 2 r50_pa           MODEL.part_attn=true MODEL.pa_parts=4 TRAIN.pa_lr=5.0e-4
launch 3 r50_pa_ctrl      MODEL.part_attn=true MODEL.pa_parts=4 MODEL.pa_depth=0 TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=0.1
wait_batch "EXP_NAME=(r50_attnp_nodrop|r50_tl_nodrop|r50_pa|r50_pa_ctrl)$"

echo "=== BATCH R: 非 mean 读出（门控 / 拼接）==="; wait_free4
RO="MODEL.tokenlearner=true MODEL.tl_dropout=0.0 MODEL.tl_replace=true MODEL.tl_ent_mode=ceiling MODEL.tl_js_w=0.0 TRAIN.tl_reg_w=0.1 TRAIN.tl_aux_w=0.5"
launch 0 r50_tl_gate_aux       $RO MODEL.tl_readout=gate
launch 1 r50_tl_gate_ceil_aux  $RO MODEL.tl_readout=gate   MODEL.tl_ent_ceiling=1.0 MODEL.tl_ent_ramp=0.2
launch 2 r50_tl_cat_aux        $RO MODEL.tl_readout=concat
launch 3 r50_tl_cat_ceil_aux   $RO MODEL.tl_readout=concat MODEL.tl_ent_ceiling=1.0 MODEL.tl_ent_ramp=0.2
wait_batch "EXP_NAME=(r50_tl_gate_aux|r50_tl_gate_ceil_aux|r50_tl_cat_aux|r50_tl_cat_ceil_aux)$"

echo "=== BATCH 2 ==="; wait_free4
launch 0 r50_pa2_ctrl MODEL.part_attn=true MODEL.pa_from=layer3 MODEL.pa_parts=4 MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_depth=0 TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0
launch 1 r50_pa3_ctrl MODEL.part_attn=true MODEL.pa_from=layer3 MODEL.pa_parts=4 MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 MODEL.pa_depth=0 TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0
launch 2 r50_camfuse3 MODEL.cam_fuse=true MODEL.cam_fuse_regions=2 TRAIN.cam_fuse_w=1.0 TRAIN.cam_margin=1.3 TRAIN.cam_margin2=1.9
launch 3 r50_hypfuse2 MODEL.hyp_fuse=true MODEL.hyp_combiner=true TRAIN.hyp_aux_w=0.2 TRAIN.cam_crop=0.0 TRAIN.cam_consistency=0.0
wait_batch "EXP_NAME=(r50_pa2_ctrl|r50_pa3_ctrl|r50_camfuse3|r50_hypfuse2)$"

echo "=== BATCH 3 ==="; wait_free4
launch 0 r50_rerank2 MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=2 MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 MODEL.part_rerank=true MODEL.pr_margin=1.3 MODEL.pr_alpha=0.2 TRAIN.cam_crop=0.0 TRAIN.cam_consistency=0.0 TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0
launch 1 r50_rerank4 MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=4 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 MODEL.part_rerank=true MODEL.pr_parts=2 MODEL.pr_use_cam=true TRAIN.cam_crop=0.0 TRAIN.cam_consistency=0.0 TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0
launch 2 r50_pa4_lr1 MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=4 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 TRAIN.pa_lr=1.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0
launch 3 r50_pa4_logit_s1 MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=4 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 MODEL.pa_logit_fuse=true MODEL.pa_no_residual=true TRAIN.pa_wd=0.05 TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0 SEED=1
wait_batch "EXP_NAME=(r50_rerank2|r50_rerank4|r50_pa4_lr1|r50_pa4_logit_s1)$"

echo "=== BATCH 4 ==="; wait_free4
launch 0 r50_partaux  MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=4 MODEL.pa_grouped=true MODEL.pa_compete=false MODEL.pa_norm_spatial=true MODEL.part_train_aux=true MODEL.pta_parts=2 TRAIN.pta_aux_w=0.2 TRAIN.pta_cons_w=0.3 TRAIN.pa_lr=5.0e-4
launch 1 r50_pa4_res  MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=4 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 MODEL.pa_res_scale=0.3 TRAIN.pa_wd=0.05 TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0
launch 2 r50_pa4_k8_s2 MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=8 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 TRAIN.pa_div_w=1.0 TRAIN.pa_aux_w=0.25 TRAIN.pa_lr=5.0e-4 SEED=2
wait_batch "EXP_NAME=(r50_partaux|r50_pa4_res|r50_pa4_k8_s2)$"

echo "=== BATCH 5 ==="; wait_free4
launch 0 r50_attnp_d0_sv  MODEL.attn_pool=true MODEL.attn_readout=cls MODEL.attn_depth=0
launch 1 r50_attnp_t49_sv MODEL.attn_pool=true MODEL.attn_readout=cls MODEL.attn_depth=1 MODEL.attn_tokens=49
wait_batch "EXP_NAME=(r50_attnp_d0_sv|r50_attnp_t49_sv)$"

echo "=== EVAL PASS ==="
for n in r50_gc r50_gc_ctrl r50_clip_gap r50_clip_ap r50_tl_ceil15_nojs r50_tl_ceil10_nojs r50_tl_ceil05_ramp r50_tl_ceil05_res r50_attnp_nodrop r50_tl_nodrop r50_attnp_d0_sv r50_attnp_t49_sv r50_pa r50_pa_ctrl r50_pa2_ctrl r50_pa3_ctrl r50_camfuse3 r50_hypfuse2 r50_rerank2 r50_rerank4 r50_pa4_lr1 r50_pa4_logit_s1 r50_partaux r50_pa4_res r50_pa4_k8_s2 r50_tl_gate_aux r50_tl_gate_ceil_aux r50_tl_cat_aux r50_tl_cat_ceil_aux; do
  d=$(ls -dt outputs/$n/*/ 2>/dev/null | head -1)
  if [ -n "$d" ] && [ -f "${d}swa_f0.5.pth" ]; then
    if [ ! -f "dumps/single_${n}_probs.npy" ]; then
      g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
      CUDA_VISIBLE_DEVICES=$g $PY scripts/test_tta.py --checkpoint "${d}swa_f0.5.pth" --views base --dump "dumps/single_${n}" > "logs/${n}_eval.log" 2>&1
      echo "  eval $n done"
      rm -f "${d}best.pth" "${d}last.pth" "${d}swa_f0.6.pth" "${d}swa_f0.7.pth"
    fi
  else
    echo "  skip $n (no ckpt)"
  fi
done
echo "ALL_DONE"
