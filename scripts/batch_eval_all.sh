#!/usr/bin/env bash
# 对一批实验的全部权重（best/last/swa_f0.5/0.6/0.7）做单视图评测。
# 用户要求：每个 SWA 平滑权重和 best、last 都要测——测试成本低，只测 f0.5 会误判。
set -u
cd "$(dirname "$0")/.." || exit 1
if [ -x /home/wpx/anaconda3/bin/python ]; then PY=/home/wpx/anaconda3/bin/python; else PY=/root/miniconda3/bin/python; fi
LOG=logs/eval_all.log
NAMES="${1:-}"
[ -z "$NAMES" ] && NAMES="r50_base_repro r50_base_e100_sv r50_pa4 r50_pa4_s1 r50_pa4_s2 r50_pa4_k8 r50_pa4_ctrl r50_pa4_logit r50_pa4_l3_k4b r50_pa2 r50_pa3 r50_pa5 r50_pa_aux r50_attnp_cls r50_attnp_ctrl r50_attnp_gate r50_attnp_cat r50_attnp_cls_d2 r50_attnp_t49 r50_tl_cls_d1 r50_tl_cls_d2 r50_tl_gate_s1 r50_tl_gate_s2 r50_tl_catgap r50_tl_gate_catgap r50_nocam r50_camfuse r50_long75 r50_se_gate_res r50_gth_pad08"
for n in $NAMES; do
  d=$(ls -dt outputs/$n/*/ 2>/dev/null | head -1)
  if [ -z "$d" ]; then echo "[$(date '+%F %T')] SKIP $n (无目录)" >> "$LOG"; continue; fi
  echo "[$(date '+%F %T')] EVAL_ALL $n" >> "$LOG"
  "$PY" scripts/eval_all_ckpts.py --run-dir "$d" >> "$LOG" 2>&1
done
echo "[$(date '+%F %T')] batch_eval_all 结束" >> "$LOG"
