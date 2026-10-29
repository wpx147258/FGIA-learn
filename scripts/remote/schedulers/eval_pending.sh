#!/usr/bin/env bash
# 评测已训练完但没自动评测的 4 格（pa4_l3_k8 / pa4_k8_s1 / tl_ent3_aux / tl_ent3_aux_res）
cd /root/autodl-tmp/cub-bird || exit 1
export HF_ENDPOINT=https://hf-mirror.com
freest_gpu () {  # 打印显存最空闲的 GPU 编号
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '
}
for n in r50_pa4_l3_k8 r50_pa4_k8_s1 r50_tl_ent3_aux r50_tl_ent3_aux_res; do
  for _ in $(seq 1 240); do
    d=$(ls -dt outputs/$n/*/ 2>/dev/null | head -1)
    if [ -n "$d" ] && [ -f "${d}swa_f0.5.pth" ]; then break; fi
    sleep 30
  done
  d=$(ls -dt outputs/$n/*/ 2>/dev/null | head -1)
  [ -f "${d}swa_f0.5.pth" ] || { echo "no ckpt $n"; continue; }
  g=$(freest_gpu)
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  if [ "$m" -gt 7000 ]; then
    echo "等待整卡 $n"; sleep 600; g=$(freest_gpu)
  fi
  echo "[$(date '+%F %T')] eval $n on GPU$g (used ${m}MiB)"
  CUDA_VISIBLE_DEVICES=$g /root/miniconda3/bin/python scripts/test_tta.py \
    --checkpoint "${d}swa_f0.5.pth" --views base --dump "dumps/single_${n}" > "logs/${n}_eval.log" 2>&1
  grep -h "总融合" "logs/${n}_eval.log" | tail -1
done
echo "PENDING_EVAL_DONE"
