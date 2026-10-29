# 远程调度脚本存档（2026-10-29）

这些脚本**原本直接写在远程 `/root/` 下、未入库**，2026-10-29 统一收回仓库存档。
执行时的路径约定：远端项目目录 `/root/autodl-tmp/cub-bird`，Python `/root/miniconda3/bin/python`，
必须 `export HF_ENDPOINT=https://hf-mirror.com`。

| 脚本 | 作用 |
|---|---|
| `restore7.sh` | 补齐链：BATCH R（门控/拼接读出）→ 2（控制臂/多区域）→ 3（rerank、pa4 变体）→ 4 → 5，末尾自动评测并清理多余权重 |
| `restore_plan.sh` / `restore3|4|6.sh` | 上面那条链的历史版本（按每次修 bug 后重写） |
| `chain_restore.sh` / `chain_e.sh` | 串联用的等待器：等第一轮名字消失后再跑下一批 |
| `eval_pending.sh` | 挑显存最空的卡，评测「已训练完但没评测」的格子 |
| `launch_backbones.sh` / `launch_backbones2.sh` | 换预训练权重的四格（a1 / tv × 有/无注意力）；v2 把 lr 统一回 1e-3 |
| `launch_plan.sh` | 原计划第一批：gc / gc_ctrl / clip_gap / clip_ap |
| `launch_ent.sh` / `launch_pa4v2.sh` / `launch_pa4k8.sh` / `launch_wt.sh` | 更早的一次性启动脚本（TL 压紧格、pa4 变体、头+翅） |

⚠️ 这些是**存档副本**，实际执行的是远端 `/root/` 下的同名文件；改动请改远端再同步回来。
