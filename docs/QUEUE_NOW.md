# 当前排队实验总表（2026-10-29 10:15）

口径：单视图、单权重（短边 512 → 中心裁 448），与 `r50_base_repro` = **89.06** 逐张配对。
所有 run 跑完自动 `test_tta --views base --dump`（`restore_plan.sh` 末尾的 EVAL PASS）。

## 口径：评价指标已换轴

| 指标 | 工具 | 判读 |
|---|---|---|
| 富集比（落在 GT 框内的质量） | `probe_mask_on_object.py` | 均匀 1.00× / GAP 1.37× |
| 逐 token CE | 训练日志 `train_logits_tl` | 每个部件能否单独分类（0.8~1.0 = 能） |
| token 特征去中心余弦 | `probe_token_stats.py` | 0 = 各部件提供不同证据；1 = 同一份 |
| 门控值分布 | `probe_gate_stats.py` | 门是否真的打开；关闭比例、|pooled−GAP| |

---

# 当前排队（2026-10-29 14:00）

## A. 正在跑

| # | 实验 | 位置 | 进度 | 目的 |
|---|---|---|---|---|
| 1 | `r50_attnp_nodrop_s1` | 本地 | ep32 val 87.57 | 89.23（dropout 对齐后的最高注意力成绩）的 **seed 复现** |
| 2 | `r50_pa2_ctrl` | 远程 GPU0 | ep21 val 85.22 | pa2（竞争 tau=1）的参数对齐控制臂 |
| 3 | `r50_pa3_ctrl` | 远程 GPU1 | ep21 val 85.07 | pa3（tau=0.1）的控制臂 |
| 4 | `r50_camfuse3` | 远程 GPU2 | ep16 val 84.76 | 三路 CAM 融合（整图 + margin1.3 + margin1.9） |
| 5 | `r50_hypfuse2` | 远程 GPU3 | ep12 val 82.32 | top1/top2 双类别区域 + 可学习融合 |

## B. 远程队列（按序，`/root/restore7.sh`）

| # | 批次 | 实验 | 目的 |
|---|---|---|---|
| 6 | BATCH 3 | `r50_rerank2` | 部件掩码裁出 → 共享骨干重编码（K=2） |
| 7 | BATCH 3 | `r50_rerank4` | 同上 K=4 + 通道分组 + `pr_use_cam` |
| 8 | BATCH 3 | `r50_pa4_lr1` | 部件分支 lr 1e-4（对照组 5e-4） |
| 9 | BATCH 3 | `r50_pa4_logit_s1` | **去残差 + logit 融合**的 seed 复现（原格 88.87） |
| 10 | BATCH 4 | `r50_partaux` | 部件辅助监督（pta_aux 0.2 + pta_cons 0.3） |
| 11 | BATCH 4 | `r50_pa4_res` | **有界残差 0.3**（10/28 那次因 res_scale bug 崩溃，从未有结果） |
| 12 | BATCH 4 | `r50_pa4_k8_s2` | **K=8 第三粒 seed**（现有 89.21 / 89.09，是唯一两粒同向为正的注意力配置） |
| 13 | BATCH 5 | `r50_attnp_d0_sv` | AttnPool depth=0 控制臂（单视图） |
| 14 | BATCH 5 | `r50_attnp_t49_sv` | 49 token 单视图（此前只有四视图 89.13） |
| 15 | EVAL PASS | 上述全部 + 已训完未评的 | 自动 `test_tta --views base`，并删除多余权重 |

## C. 本地队列（按序，watcher 链）

| # | 实验 | 目的 |
|---|---|---|
| 16 | `r50_attnp_gate` | **AttnPool 门控读出**（`GAP + Σ g_n·out(t_n)`，初始输出精确为 0）；判断 196 位置的门会不会像 8 区域那样被关死 |
| 17 | `r50_attnp_cls_rep` | **替换 GAP 的 AttnPool**（89.23 同配置：CLS、depth1、dropout 0；初始 a=0.5，|pooled−GAP|/|GAP|≈2.3、logit 差 0.058） |

## D. 已完成、已有结论（今日）

`gc` 88.87 / `gc_ctrl` 88.75 / `clip_gap` 84.47 / `clip_ap` 80.51 /
`attnp_nodrop` 89.23 / `tl_nodrop` 88.68 / `pa` 88.80 / `pa_ctrl` 88.68 /
F 轮 `ceil15` 84.71、`ceil10` 84.05、`ceil05_ramp` 84.41、`ceil05_res` 88.59 /
BATCH R `gate` 89.09、`gate_ceil` 86.76、`cat` 88.09、`cat_ceil` 84.59 /
`attnp_cls_d2` 88.95 / `tl_cls_d2` 88.63 / `pa4_l3_k8` 88.71 / `pa4_l3_k4` 88.63 /
`pa4_k8_s1` 89.09 / `se_gate_res` 88.85 / `se_gate_sigmoid` 88.78 / `se_fixed` 87.87

## E. 仍未排 / 待决策

| 项 | 卡在哪 |
|---|---|
| `r50_rerank2_ctrl` | 原队列里与 `rerank2` 参数逐字相同（复制粘贴漏改），需重新定义对照 |
| `r50_s8_gc` | 条件项：`gc` 无收益，暂不跑 |
| PMG **200 epoch** | 计划 H3，只有 50 epoch 的 89.25；约 7 小时 |
| Cross-X / DF-GMM / NAC-DFL | 仓库无实现，需先写代码 |
| `r50_gth_w_s3` | 强监督项，按用户要求暂停 |
