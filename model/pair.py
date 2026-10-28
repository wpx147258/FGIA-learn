"""成对判别（API-Net, Zhuang/Wang/Qiao, AAAI 2020）。

为什么加它：我们的错误不是"没看到"，而是"两个都合理"。
诊断里 8 个类的类中心余弦 > 0.9，占测试集 7% 的图片、贡献 22.5% 的错误，
而且这些类在训练集上都是 100%。单图分类器在这类对偶上没有任何额外信息可用。

机制（对应原文 Eq.1-10）：

    x_m      = MLP([x_i ; x_j])          两张图拼起来，归纳成「这对的差异在哪」
    g_i      = sigmoid(x_m ⊙ x_i)        拿差异向量当探针，指出 x_i 里哪些通道有线索
    x_i^self = x_i + x_i ⊙ g_i           自己的门控
    x_i^other= x_i + x_i ⊙ g_j           对家的门控（用别人的眼睛看自己）

每对产出四个特征，都过**同一个**分类器 W。self 特征应该比 other 特征更准，
所以加一个排名铰链损失把两者的优先级推开。

三个关键性质：
  1. 配对发生在池化特征之后，不增加前向次数（对比 cam_crop 是两倍前向）
  2. 测试时整套卸载，单图走 backbone → W，和现在完全一样
  3. 配对是批内按特征欧氏距离选的最近邻，不是随机配 —— 所以它挑中的是
     当前模型眼里最像的那一对，正是我们要它学的

和原论文的差异（有意为之，写在文档里）：
  - 原论文用 30 类 × 4 图 = 120 的批；我们 12~24G 卡在 448 下装不下这么大，
    用 TRAIN.pair_p × TRAIN.pair_k 配置，默认 8 × 4 = 32。
  - 原论文主损失就是四个特征的 CE；我们**保留**主分支自己的 CE（TRAIN.pair_main_w），
    这样和 89.10 的配方只差"多了一项"，对比是单变量的。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PairAux(nn.Module):
    """批内配对 + 门控交互。挂在 Classifier 上，只在训练时生效。

    参数只有 mutual vector 那个 MLP（2C → hidden → C），ResNet-50 上是
    4096×512 + 512×2048 ≈ 3.1M。门控和四个增强特征都不带参数，
    分类器复用主分支的 head。
    """

    def __init__(self, feat_dim: int, hidden: int = 512):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_dim),
        )

    @staticmethod
    def _nearest(dist: torch.Tensor, same: torch.Tensor):
        """在批内找同类最近邻、异类最近邻。

        dist [B,B] 是欧氏距离，same [B,B] 是 bool 的同类矩阵。
        对角线和不符合条件的格子填 inf，argmin 自然跳过。
        约束：每类至少 2 张（由 P×K 采样保证），每批至少 2 个类。
        """
        B = dist.size(0)
        eye = torch.eye(B, dtype=torch.bool, device=dist.device)
        same = same & ~eye
        diff = (~same) & ~eye
        i_same = dist.masked_fill(~same, float("inf")).argmin(dim=1)
        i_diff = dist.masked_fill(~diff, float("inf")).argmin(dim=1)
        return i_same, i_diff

    def forward(self, model, pooled: torch.Tensor, label: torch.Tensor) -> dict:
        """pooled [B,C] 是主分支的池化特征；label [B] 是主标签。

        返回的 logits 和 batch 内的下标，标签留给 criterion 去取
        （mixup 下要用软目标，模型这层不该知道软目标长什么样）。
        """
        B = pooled.size(0)
        # 距离只在选邻居时用，不参与梯度；转 fp32 避免 bf16 下 cdist 的精度问题
        with torch.no_grad():
            f = pooled.detach().float()
            same = label.view(-1, 1).eq(label.view(1, -1))
            dist = torch.cdist(f, f)
            i_same, i_diff = self._nearest(dist, same)

        # 每张图出两个对：(自己, 同类最近邻) 和 (自己, 异类最近邻)
        # -> 2B 个有序对
        partner = torch.stack([i_same, i_diff], dim=1)          # [B,2]
        idx_i = torch.arange(B, device=pooled.device).repeat_interleave(2)
        idx_j = partner.reshape(-1)

        xi = pooled[idx_i]
        xj = pooled[idx_j]

        xm = self.mlp(torch.cat([xi, xj], dim=1))                # 差异向量
        g_i = torch.sigmoid(xm * xi)
        g_j = torch.sigmoid(xm * xj)

        head = model.head
        out = {
            "pair_self_i": head(xi + xi * g_i),
            "pair_other_i": head(xi + xi * g_j),
            "pair_self_j": head(xj + xj * g_j),
            "pair_other_j": head(xj + xj * g_i),
            "pair_idx_i": idx_i,
            "pair_idx_j": idx_j,
        }
        # 门控的稀疏度：如果 g 全饱和到 0 或 1，门控就退化成常量，
        # 这个数和当初诊断 CAP 的 tanh 饱和用的是同一套思路
        out["pair_gate_mean"] = g_i.detach().mean()
        return out
