# 排队与现场（2026-10-30 02:10）

> 上下文压缩前更新。**每次派批次/改链路后必须同步这份**。

## 0. 口径（两条硬约定）

1. **主口径 = 全训（5,994）**：用户 10-29 22:40 决定「后面不切 val，直接全训」。
   文献核查支持：MA-CNN/NTS-Net/API-Net/CAL/WS-DAN/Mask-CNN 都是 5,994 训练、不切 val、报 test。
   ⚠️ 全训后日志里的 `val_top1` 是**训练集**准确率（96~100%），不能用于选模型，只能按固定规则评测。
2. **每个 run 都要评 5 个权重**：`best` / `last` / `swa_f0.5` / `f0.6` / `f0.7`
   （工具 `scripts/eval_all_ckpts.py`）。**主表固定 f0.5**，其余只作稳健性检查。

## 1. 全训结果总表（f0.5 单视图，5,794 test）

| 实验 | test | vs 基线 | 说明 |
|---|---|---|---|
| 历史基线 4 粒 | 89.94 / 89.92 / 89.70 / 89.64（均 **89.80**） | — | 更早的代码 |
| `all_base` | 89.61 | — | 当前代码，单粒 |
| **`all_pa4k8`** | **89.75** | **+0.14** | **K=8 / compete / 无 JS：PA 最好** |
| `all_attnpcls` | 89.85（S42）/ 89.52（S1） | +0.24 / −0.09 | **+0.24 未复现** |
| `all_pa4` | 89.58 | −0.03 | K=4 / compete / 无 JS |
| `all_pa4k8_blur` | 89.59 | −0.02 | ⚠️ 与 `all_pa4k8` **同有效配置**（blur 没生效） |
| `all_pa4_blur` | 89.42 | −0.19 | ⚠️ 同上，是 `all_pa4` 的重跑 |
| `all_pa4js` | 89.13 | −0.48 | K=4 / compete / **JS** |
| `all_pa4js_blur` | 89.13 | −0.48 | ⚠️ 同上，blur 没生效 |
| `all_sig_js` | 89.23 | −0.38 | K=4 / **无 compete** / JS |
| `all_sig` | 89.02 | −0.59 | K=4 / 无 compete / 无 JS |
| `all_sig_blur` | 89.02 | −0.59 | ⚠️ 与 `all_sig` 五个权重**逐位相同** |

## 2. 从这张表读出的 PA 现状（2026-10-30 02:00 修正）

**旧结论作废**：之前写「compete=false 比 compete=true 好」「掩码平滑修好形态但不涨点」，两条都错。

| 对照（K=4，全训，f0.5） | 差 |
|---|---|
| compete 的效果（去掉 JS 后）：89.58 vs 89.02 | **compete +0.56** |
| JS 的效果（固定 compete）：89.13 vs 89.58 | **JS −0.45** |
| K 的效果（compete、无 JS）：K=8 89.75 vs K=4 89.58 | **K ↑ +0.17** |

也就是说：**"compete 让格子互斥过强"这个假设，在把 JS 控制住之后不成立**；真正在动的是
「**去掉 JS**」和「**K 放大**」。JS 是唯一一个明确有害的项（−0.45，约为噪声的 3 倍）。

### 掩码平滑（blur）格子全部无效——实现 bug

`pa_mask_blur` 只写在 `part_masks()` 里（`mul` 路径和探针调用），`forward()` 内联重算掩码时
没有平滑 ⇒ `pa_mask_blur=true` 对训练**完全无效**。证据：`all_sig` 与 `all_sig_blur` 的
best/last/f0.5/f0.6/f0.7 = 84.73/89.07/89.02/89.01/88.95，**五个完全相同**，两份 config 只差这一项。
修复提交 **9cf2531**。所以：

- 四个 `*_blur` 格子 = 各自非 blur 孪生格的**重跑**；它们之间 0.00~0.16 的差就是**跑动噪声地板**；
- 「邻居相关 0.27→0.66 但 test 不动」这条因果**从来没被测过**（探针测的是模型没用过的掩码）；
- 真正测平滑的是 chain_gap G3 的 `all_pa4js_blur_v2`（修复后的代码）。

## 3. 正在跑（2026-10-30 02:45）

| 位置 | 实验 | 批次 | 动机 |
|---|---|---|---|
| 远程 GPU0–2 | `all_pa4k16` / `all_pa4k8_s7` / `all_pa4k8_l3` | `chain_pa.sh` | K 往上、89.79 的第二粒、layer3 |
| 远程 GPU3 | `all_parand8` | 手动（02:29 重启） | **固定随机掩码对照**：掩码到底学到东西没有 |
| 本地 3090 | `pa4js_repro2` / `e75m2_s7`（ep41） | — | 可复现性测试 / 长训练 |
| 本地排队 | `all_base_s1` → `all_pa4k8_s1` | `queue_all_base_s1.sh`、`queue_local_pa4k8_s1.sh` | 同机同 SEED=1 的基线 vs PA 单变量对照 |

## 4. 之后自动接的批次（远程，已挂链）

`chain_pa → chain_long → chain_gap → chain_true_repro → chain_res → chain_restore`

| 顺序 | 脚本 | 内容 |
|---|---|---|
| 下一批 | `chain_long.sh` | `all_e75` / `all_e75_m2` / `all_e100` / `all_attnpcls_e75`（破 90 主押注） |
| 再下 | `chain_gap.sh` | 12 格缺口：t49 / attnp_d0 / pa4k8_cls / **pa4k8_ent1** / **pa4k8_ent3** / aux1 / pos / dim512 / heads8 / div03 / div30 / pa4js_blur_v2 |
| 再下 | `chain_true_repro.sh` | `r50_pa4js_c1`（std 分母 195 的精确复现） |
| 再下 | `chain_res.sh` | **新实验批**：分辨率 512（基线 / PA）、CAM 裁剪权重 0.35、一致性 0.6 |
| 最后 | `chain_restore.sh` | **补回被重排换下的五格**：tau005 / tau02 / sigk8 / sigk8_blur / pa4js_l3，外加 k2 与 `attnp_mean` |

### chain_res：按已有结论挑的四格

| 实验 | 依据 |
|---|---|
| `all_r512_base` | **输入分辨率从没动过**（全程 448）。细粒度上这是标准增益项；而 stride-8 试过更差（88.56），说明"改分辨率"≠"改下采样" |
| `all_r512_pa4k8` | 同上 + 目前最好的 PA 配置 |
| `all_cam035` | CAM 裁剪分支是**唯一被反复验证为正**的组件（`nocam` 88.21 vs 89.06，+0.85~1.19），但只扫过 margin/alpha，**损失权重从没扫过** |
| `all_cons06` | 同一分支的一致性权重 0.3→0.6 |

### 为什么要 chain_restore

前面几轮重排（把 chain_pa 插到 chain_long 前、用熵上限替换 τ 扫描、本地 k2 换成基线复现）
换下来 5 个已经排好的实验。这些不是失败，只是被我挪走了——现在原样补回到整条链的最后，
一格都不丢，也不会打乱已经在跑的批次。

## 5. 机制诊断（仍然有效的那部分）

| 发现 | 数值 | 出处 |
|---|---|---|
| 部件掩码是**铺满全图的碎斑** | 熵 4.45–4.96 / 5.28、平均 **13 个连通块**、邻居相关 0.27 | 探针 |
| **compete 把部件变成随机划分** | 每位置分配熵 0.55/1.386、独占度 0.65；Σ_k m ≡ 1 | 2026-10-29 23:50 |
| 加性残差在 logit 空间是**类无关偏置** | cos(Δlogit, logit) = −0.995；去掉类间均值后只剩 3.6% | `probe_part_residual.py` |
| 富集比 | pa4_js 1.12×、pa4 1.13×（GAP 1.37×） | `probe_mask_on_object.py` |
| 乘性残差 γ 训完仍很小 | 四个乘性变体 88.78–88.99 | 2026-10-29 23:10 |

⚠️ 上面「碎斑」那组数是在**未修复的代码**上测的，形态本身不受影响（`part_masks` 一直是同一套），
但「平滑能修好它」这条推论要等 `pa4js_blur_v2` 才能下。

## 6. pa4_js 的同 seed 复现（5,395 口径）

| run | 机器 | SEED | best val | test f0.5 |
|---|---|---|---|---|
| 原 `pa4_js` | 远程 3080Ti | 42 | **89.91** | 88.82 |
| `pa4js_s42r` | 远程 3080Ti | 42 | 89.76 | 88.56 |
| `pa4js_s42` | 本地 3090 | 42 | 88.97 | 88.78 |
| `pa4js_r2` | 本地 3090 | 7 | 89.29 | 88.97 |
| `pa4js_repro2` | 本地 3090 | 42 | 跑中 | — |

**跑动噪声地板（全训口径实测）**：同有效配置的两粒差 **0.00~0.16**
（`sig` vs `sig_blur` = 0.00；`pa4` vs `pa4_blur`、`pa4k8` vs `pa4k8_blur` = 0.16）。
⇒ **任何 <0.2 的单粒差一律写「未检出」**，这条现在是实测数字，不是经验值。

## 7. 脚本与文件位置

- **远程** `/root/`：`chain_sig.sh`、`chain_cls.sh`、`chain_pa.sh`、`chain_long.sh`、`chain_gap.sh`、
  `chain_true_repro.sh`、`eval_missing.sh`、`eval_pa4_old.sh`（较新的几个已同步进
  `scripts/remote/schedulers/`）
- **本地** `scripts/`：`queue_all_base_s1.sh`、`queue_all_e75m2_s7.sh`、`queue_pa4js_repro_test.sh`、
  `watch_repro.sh`、`eval_all_ckpts.py`、`batch_eval_all.sh`
- **探针**：`probe_mask_on_object.py`（富集比）、`probe_part_residual.py`、`probe_gate_stats.py`、
  `probe_token_stats.py`、`vis_token_maps.py`
- **文档**：`docs/RUNNING_NOW.md`、`docs/EXPERIMENT_HISTORY.md`、`EXPERIMENT_INVENTORY.md`、
  `AUTO_STATE.md`、`PART_ATTENTION_FINDINGS.md`

## 8. 判读规则

- 单粒差 <0.2 ⇒ 「未检出」；要下结论必须 ≥3 粒同向；
- 主表固定 `swa_f0.5`，不按 test 挑权重；
- 旧的 `*_blur` 四格**不能**再当「平滑无效」的证据（实现 bug），只能当重跑噪声样本。
