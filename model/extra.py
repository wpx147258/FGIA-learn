"""参数匹配的「额外加性通路」控制臂（2026-10-31）。

为什么需要它：推理期因果干预（probe_part_causal2.py）把部件注意力分支的贡献拆成
  · 去掉整条分支：−0.28（89.35 → 89.07）
  · 掩码换成均匀/平移：0 张翻转
也就是说**分支的价值全部来自"多了一条参数化的加性通路"，掩码内容贡献为 0**。

那么正确的对照是一个**没有任何空间结构、只是多一层加性 MLP**的分支：
如果它也能拿到 +0.28，就证明部件注意力 = 额外容量，空间结构完全没有贡献；
如果它明显更差，说明 PA 的通道分组/池化结构仍有价值（只是与掩码无关）。

结构：pooled + Linear(GELU(Linear(pooled)))，输出层零初始化 ⇒ 初始与基线逐位等价，
和 PartAttention / AttnPool 的出口约定一致。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ExtraBranch(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        # 输出层零初始化 ⇒ 初始与基线逐位等价（与 PartAttention / AttnPool 的约定一致）
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return pooled + self.net(pooled)
