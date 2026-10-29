#!/usr/bin/env bash
# 补齐 10/27 原计划里从未执行的实验（分三轮，每轮 4 卡）
# 第一轮：部件注意力的控制臂与消融（原队列 1/2/3 与 pa2/pa3 的对照）
# 第二轮：多区域（三路 CAM 融合、hyp_fuse2、部件裁出重编码 rerank2/4）
# 第三轮：pa4 的学习率与 logit 融合 seed、部件辅助监督
cd /root/autodl-tmp/cub-bird || exit 1
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/miniconda3/bin/python
COMMON="--config configs/baseline_config.yaml --set DATASET.image_size=448 TRAIN.epochs=50 TRAIN.batch_size=16 TRAIN.mixup=0.2 TRAIN.cutmix=1.0 TRAIN.mixup_p=0.7 TRAIN.label_smoothing=0.05 TRAIN.cam_crop=0.2 TRAIN.cam_alpha_crop=0.2 TRAIN.cam_margin=1.3 TRAIN.cam_consistency=0.3 TRAIN.cam_consistency_T=2.0 TRAIN.backbone_lr_layers.stem=0.01 TRAIN.backbone_lr_layers.layer1=0.01 TRAIN.backbone_lr_layers.layer2=0.03 TRAIN.backbone_lr_layers.layer3=0.05 TRAIN.backbone_lr_layers.layer4=0.05 TRAIN.lr=1.0e-3 MODEL.name=resnet50.fb_swsl_ig1b_ft_in1k MODEL.dropout=0.1 TRAIN.swa_start_frac=[0.5,0.6,0.7]"
wait_gpu() {  # $1 = index
  for _ in $(seq 1 360); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1")
    [ "$m" -lt 2000 ] && sleep 20 && return 0
    sleep 20
  done; return 1
}
go() { local gpu=$1 name=$2; shift 2
  wait_gpu $gpu || return 1
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY train.py $COMMON "$@" EXP_NAME="$name" > "logs/${name}_train.log" 2>&1 &
  echo "launch $name on gpu$gpu pid $!"
}
echo "=== PHASE F: 熵上限（局部性），不要求分离 ==="
TL="MODEL.tokenlearner=true MODEL.tl_dropout=0.0 MODEL.tl_ent_mode=ceiling MODEL.tl_js_w=0.0 TRAIN.tl_reg_w=0.1 TRAIN.tl_aux_w=0.5"
go 0 r50_tl_ceil15_nojs  $TL MODEL.tl_replace=true MODEL.tl_ent_ceiling=1.5 &
go 1 r50_tl_ceil10_nojs  $TL MODEL.tl_replace=true MODEL.tl_ent_ceiling=1.0 &
go 2 r50_tl_ceil05_ramp  $TL MODEL.tl_replace=true MODEL.tl_ent_ceiling=0.5 MODEL.tl_ent_ramp=0.2 &
go 3 r50_tl_ceil05_res   $TL MODEL.tl_ent_ceiling=0.5 &
wait
echo "=== PHASE E: 注意力对齐臂 ==="
go 0 r50_attnp_t49_sv MODEL.attn_pool=true MODEL.attn_readout=cls MODEL.attn_depth=1 MODEL.attn_tokens=49 &
go 1 r50_attnp_d0_sv  MODEL.attn_pool=true MODEL.attn_readout=cls MODEL.attn_depth=0 &
go 2 r50_attnp_nodrop MODEL.attn_pool=true MODEL.attn_readout=cls MODEL.attn_depth=1 MODEL.attn_dropout=0.0 &
go 3 r50_tl_nodrop    MODEL.tokenlearner=true MODEL.tl_readout=cls MODEL.tl_attn_depth=1 MODEL.tl_dropout=0.0 &
wait
echo "=== PHASE B: 控制臂 ==="
go 0 r50_pa       MODEL.part_attn=true MODEL.pa_parts=4 TRAIN.pa_lr=5.0e-4 &
go 1 r50_pa_ctrl  MODEL.part_attn=true MODEL.pa_parts=4 MODEL.pa_depth=0 TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=0.1 &
go 2 r50_pa2_ctrl MODEL.part_attn=true MODEL.pa_from=layer3 MODEL.pa_parts=4 MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_depth=0 TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0 &
go 3 r50_pa3_ctrl MODEL.part_attn=true MODEL.pa_from=layer3 MODEL.pa_parts=4 MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 MODEL.pa_depth=0 TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0 &
wait
echo "=== PHASE C: 多区域 ==="
go 0 r50_camfuse3  MODEL.cam_fuse=true MODEL.cam_fuse_regions=2 TRAIN.cam_fuse_w=1.0 TRAIN.cam_margin=1.3 TRAIN.cam_margin2=1.9 &
go 1 r50_hypfuse2  MODEL.hyp_fuse=true MODEL.hyp_combiner=true TRAIN.hyp_aux_w=0.2 TRAIN.cam_crop=0.0 TRAIN.cam_consistency=0.0 &
go 2 r50_rerank2   MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=2 MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 MODEL.part_rerank=true MODEL.pr_margin=1.3 MODEL.pr_alpha=0.2 TRAIN.cam_crop=0.0 TRAIN.cam_consistency=0.0 TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0 &
go 3 r50_rerank4   MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=4 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 MODEL.part_rerank=true MODEL.pr_parts=2 MODEL.pr_use_cam=true TRAIN.cam_crop=0.0 TRAIN.cam_consistency=0.0 TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0 &
wait
echo "=== PHASE D: 变体 ==="
go 0 r50_pa4_lr1     MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=4 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 TRAIN.pa_lr=1.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0 &
go 1 r50_pa4_logit_s1 MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=4 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 MODEL.pa_logit_fuse=true MODEL.pa_no_residual=true TRAIN.pa_wd=0.05 TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0 SEED=1 &
go 2 r50_partaux     MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=4 MODEL.pa_grouped=true MODEL.pa_compete=false MODEL.pa_norm_spatial=true MODEL.part_train_aux=true MODEL.pta_parts=2 TRAIN.pta_aux_w=0.2 TRAIN.pta_cons_w=0.3 TRAIN.pa_lr=5.0e-4 &
go 3 r50_pa4_res MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=4 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 MODEL.pa_res_scale=0.3 TRAIN.pa_wd=0.05 TRAIN.pa_lr=5.0e-4 TRAIN.pa_aux_w=0.5 TRAIN.pa_div_w=1.0 &
go 3 r50_pa4_k8_s2 MODEL.part_attn=true MODEL.pa_from=layer4 MODEL.pa_parts=8 MODEL.pa_grouped=true MODEL.pa_compete=true MODEL.pa_norm_spatial=true MODEL.pa_tau=0.1 TRAIN.pa_div_w=1.0 TRAIN.pa_aux_w=0.25 TRAIN.pa_lr=5.0e-4 SEED=2 &
wait
echo "=== EVAL PASS ==="
for n in r50_gc r50_gc_ctrl r50_clip_gap r50_clip_ap r50_tl_ceil15_nojs r50_tl_ceil10_nojs r50_tl_ceil05_ramp r50_tl_ceil05_res r50_tl_ceil05_noramp r50_tl_ceil03_noramp r50_attnp_nodrop r50_tl_nodrop r50_attnp_d0_sv r50_attnp_t49_sv r50_pa r50_pa_ctrl r50_pa2_ctrl r50_pa3_ctrl r50_camfuse3 r50_hypfuse2 r50_rerank2 r50_rerank4 r50_pa4_lr1 r50_pa4_logit_s1 r50_partaux r50_pa4_res r50_pa4_k8_s2; do
  d=$(ls -dt outputs/$n/*/ 2>/dev/null | head -1)
  if [ -n "$d" ] && [ -f "${d}swa_f0.5.pth" ]; then
    if [ ! -f "dumps/single_${n}_probs.npy" ]; then
      "$PY" scripts/test_tta.py --checkpoint "${d}swa_f0.5.pth" --views base --dump "dumps/single_${n}" > "logs/${n}_eval.log" 2>&1
      echo "eval $n done"
      d2=$(dirname "$d")
      rm -f "${d2}/best.pth" "${d2}/last.pth" "${d2}/swa_f0.6.pth" "${d2}/swa_f0.7.pth"
    fi
  else
    echo "skip $n (no ckpt)"
  fi
done
echo "ALL_PHASES_DONE"
