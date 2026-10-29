# 排队与现场（2026-10-30 00:40）

> 上下文压缩前更新。**每次派批次/改链路后必须同步这份**。

## 0. 口径（两条硬约定）

1. **主口径 = 全训（5,994）**：用户 10-29 22:40 决定「后面不切 val，直接全训」。
   文献核查支持：MA-CNN/NTS-Net/API-Net/CAL/WS-DAN/Mask-CNN 都是 5,994 训练、不切 val、报 test；
   我们的 5,395/599 是 VPT(ECCV'22) 那套。
   ⚠️ 全训后日志里的 `val_top1` 是**训练集**准确率（96~100%），不能用于选模型，只能按固定规则评测。
2. **每个 run 都要评 5 个权重**：`best` / `last` / `swa_f0.5` / `f0.6` / `f0.7`
   （工具 `scripts/eval_all_ckpts.py --run-dir <dir>`）。**主表固定 f0.5**，其余只作稳健性检查。

## 1. 正在跑

| 位置 | 实验 | 批次 | 动机 |
|---|---|---|---|
| 远程 GPU0–3 | `all_sig` / `all_sig_js` / `all_pa4js`（补跑）/ `all_sig_blur` | `chain_sig.sh` | **compete=false**（允许重叠）× 有无 JS × 平滑 |
| 本地 3090 | `r50_all_attnpcls_s1`（ep17） | `queue_all_attnpcls_s1.sh` | 全训最好格 89.85 的**种子复现** |

## 2. 之后自动接的批次（远程，已挂链）

| 顺序 | 脚本 | 四格 | 目的 |
|---|---|---|---|
| 下一批 | `chain_cls.sh` | `all_pa4_cls` / `all_pa4js_cls` / `all_sig_cls` | **PartAttention 的 CLS 读出**（历史只有 mean） |
| 再下一批 | `chain_long.sh` | `all_e75` / `all_e75_m2` / `all_e100` / `all_attnpcls_e75` | **全训 + 更长训练**；并把历史 `long75` 的「长度」与「双尺度一致性」拆开 |

链路：`chain_sig` →（`CHAIN_SIG_DONE`）→ `chain_cls` →（`CHAIN_CLS_DONE`）→ `chain_long`。

## 3. 已出的结果（全训协议，f0.5 单视图）

| 实验 | test | vs 全训基线 89.61 |
|---|---|---|
| `all_base` | **89.61** | — |
| **`all_attnpcls`** | **89.85** | **+0.24**（待种子复现，本地在跑） |
| `all_pa4k8_blur` | **89.59** | −0.02 |
| `all_pa4_blur` | 89.42 | −0.19 |
| `all_pa4js_blur` | 89.13 | −0.48 |
| `all_pa4k8` / `all_pa4` | 待补评 | 之前 OOM |

**掩码平滑（碎斑→区域）没有带来增益**：`pa4_blur` 89.42、`pa4js_blur` 89.13，都低于
未平滑的 `all_pa4`/`all_pa4k8`（待补评）与基线。

## 4. pa4_js 的三次运行（5,395 口径）——val 89.91 的结论

| run | 机器 | SEED | best val | test f0.5 |
|---|---|---|---|---|
| 原 `pa4_js` | 远程 3080Ti | 42 | **89.91** | 88.82 |
| `pa4js_s42r` | 远程 3080Ti | 42 | **89.76** | **88.56** |
| `pa4js_s42` | 本地 3090 | 42 | 88.97 | 88.78 |
| `pa4js_r2` | 本地 3090 | 7 | 89.29 | 88.97 |

**两条结论**：
1. **val 89.9 在远程可复现（89.76），本地复现不了（88.97）**——同种子同配置，val 差 0.79，
   而 test 只差 0.22。说明 **val（599 张）对训练数值差异极其敏感，会放大成 1 个点的假信号**。
2. **val 89.76 → test 88.56**。高 val 完全不预示高 test。pa4_js 的 test 四粒在 88.56–88.97，
   均低于基线三粒均值 88.95。

## 5. 机制诊断（回答「为什么不行」的关键证据）

| 发现 | 数值 | 出处 |
|---|---|---|
| 部件掩码是**铺满全图的碎斑** | 熵 4.45–4.96 / 5.28、平均 **13 个连通块**、邻居相关 0.27 | `figs/pa4js_maps.png` + 探针 |
| **compete 把部件变成随机划分** | 每位置在 K 个部件上的分配熵 0.55/1.386、独占度 0.65；Σ_k m ≡ 1 ⇒ 部件不能重叠 | 2026-10-29 23:50 |
| 加性残差在 logit 空间是**类无关偏置** | cos(Δlogit, logit) = −0.995；去掉类间均值后只剩 3.6% | `probe_part_residual.py` |
| 富集比 | pa4_js 1.12×、pa4 1.13×（GAP 1.37×） | `probe_mask_on_object.py` |
| 乘性残差 γ 训完仍很小 | 四个乘性变体 88.78–88.99，低于加性 89.13/89.14 | 2026-10-29 23:10 |

**当前最有解释力的因果链**：检测器是 1×1 卷积 + 逐格标准化 +（compete）逐位置 softmax，
没有任何空间先验 ⇒ 掩码是「随机的 1/K 划分」⇒ 加权平均≈GAP ⇒ 残差只是逐通道常数偏移。
掩码平滑（邻居相关 0.27→0.66）修好了形态，但 **test 没有跟上**（89.42/89.13），
说明「部位成形」不等于「部位提供信息」。

## 6. 仍未排队（按价值排序）

| # | 缺口 | 备注 |
|---|---|---|
| 1 | 全训 + 49 token | 5,395 口径 88.85 |
| 2 | 全训 + AttnPool depth=0 控制臂 | 没有它说不清 89.85 是注意力还是那层池化 |
| 3 | PA 的 K=8 + CLS 读出 | CLS 三格都是 K=4 |
| 4 | τ 扫描（0.05 / 0.2） | 掩码近 one-hot，τ 从未扫过 |
| 5 | 逐部件 CE 权重 1.0 | 只用过 0.25/0.5 |
| 6 | `attn_pos=true` | 从未跑；PART 消融里位置编码 +0.6~0.7 |
| 7 | `attn_dim=512` / `heads=8` | 256/4 是拍的 |
| 8 | 散度项权重 0.3/3.0 | 固定 1.0 |

## 7. 脚本与文件位置

- **远程** `/root/`：`chain_sig.sh`、`chain_cls.sh`、`chain_long.sh`、`launch_blur.sh`、
  `launch_alltrain.sh`、`remote_pa4js_s42.sh`、`eval_pending.sh`
  （已全部归档进 `scripts/remote/schedulers/`）
- **本地** `scripts/`：`queue_pa4js_s42.sh`、`queue_all_attnpcls_s1.sh`、
  `queue_pa4js.sh`、`batch_eval_all.sh`、`eval_all_ckpts.py`
- **探针**：`probe_mask_on_object.py`（富集比）、`probe_part_residual.py`（残差是否影响决策）、
  `probe_gate_stats.py`（门控）、`probe_token_stats.py`（掩码统计）、`vis_token_maps.py`（可视化）
- **文档**：`docs/RUNNING_NOW.md`（现场）、`docs/EXPERIMENT_HISTORY.md`（历史总表）、
  `EXPERIMENT_INVENTORY.md`（滚动清单）、`AUTO_STATE.md`（时间线日志）、
  `PART_ATTENTION_FINDINGS.md`（机制结论）

## 8. 缺口补齐批次（2026-10-30 00:50 已排队）

`/root/chain_gap.sh`：等 `CHAIN_LONG_DONE` 之后，分三批跑完 §6 的全部缺口（全训协议）。

| 批 | GPU | 实验 | 对应缺口 |
|---|---|---|---|
| **G1** | 0 | `all_t49` | 49 token |
| | 1 | `all_attnp_d0` | AttnPool depth=0 控制臂 |
| | 2 | `all_pa4k8_cls` | K=8 + CLS 读出 |
| | 3 | `all_tau005` | τ=0.05 |
| **G2** | 0 | `all_tau02` | τ=0.2 |
| | 1 | `all_aux1` | 逐部件 CE 权重 1.0 |
| | 2 | `all_pos` | `attn_pos=true`（位置编码） |
| | 3 | `all_dim512` | `attn_dim=512` |
| **G3** | 0 | `all_heads8` | `attn_heads=8` |
| | 1 | `all_div03` | 散度权重 0.3 |
| | 2 | `all_div30` | 散度权重 3.0 |

每批结束自动 `eval_all_ckpts.py` 评五个权重。链路顺序：
`chain_sig → chain_cls → chain_long → chain_gap`（全部完成约 5–6 小时）。

## 9. 「同 seed 为什么没复现 89.91」的结论与最终验证（2026-10-30 01:10）

**先纠正一处事实**：原 `pa4_js`（02:35）用的 `raw.std(dim=-1, keepdim=True)` 是 torch 默认
**correction=1，即分母 N−1 = 195**；03:25 的审计改成 `correction=0`（分母 **N = 196**）。
所以现在的代码与原 run 差的是 195 vs 196，检测器分数整体缩放 0.26%。

- 配置逐字段 diff：**只有新增的默认键，实质一致**；
- 同机器（3080Ti）+ 同种子 42 + 差这一行代码：val 曲线从 **ep1 就分叉**（73.19 vs 71.85），
  best val 89.91 vs 89.76，test 88.82 vs 88.56；
- 同代码 + 换机器（3090）+ 同种子：best val 88.97，test 88.78。

**结论**：同 seed 只固定随机数流，不固定算术。代码改动或换卡都会改变每步数值，而
AdamW 的 m/√v 归一化 + 全局梯度裁剪会把差异放大（裁剪范数里含部件分支梯度）。

**最终验证已排队** `/root/chain_true_repro.sh`（等 `CHAIN_GAP_DONE` 后跑）：
`r50_pa4js_c1` = 同机器 3080Ti + SEED=42 + **`pa_std_correction=1`（分母 195）**。
- 若它复现 89.9x 的 val 与 88.8x 的 test → 差异确认来自那一行代码；
- 若仍不同 → 训练本身在该尺度上不确定，val 的 0.5 量级差异一律不可信。

新增配置项：`MODEL.pa_std_correction`（0=N，1=N−1，**仅用于复现历史，不是调参项**）。
