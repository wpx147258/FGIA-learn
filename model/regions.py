"""多区域重编码 + 学习式路由。

动机（详见 PART_ATTENTION_FINDINGS.md 第八节）：
  * 零训练诊断里，唯二上限明显高于基线的都是「多区域重编码」：
        GT 头 + GT 身体 + 整图，三路 oracle 91.85；
        整图 + top-1/top-2 类别的 CAM 区域，oracle 91.39（top-2 单独救回 131 张）。
  * 但所有固定融合（β 加权、概率平均、置信度门控）都拿不到这部分信息
    （hypfuse 87.99 / camfuse 88.82，均低于各自的主分支）。
  * 文献（learning-to-defer：Mozannar & Sontag 2020、Narasimhan 2022、Jitkrittum 2023）
    指出正确做法是**逐样本**比较「区域对」与「主分支对」的概率，用 OOF 数据学一个
    gate，而不是固定阈值。

本模块提供三样东西：
  1. `propose_regions`  —— 从主分支特征图取 top-k 类别的 CAM 框（紧凑、类判别）；
  2. `RegionRouter`     —— 逐样本的软路由：输入是主分支与各区域的置信度/一致性统计，
                           输出 K+1 个权重（含一个「都不用」的选项）；
  3. `fit_router_oof`   —— 用途见 scripts/fit_router_oof.py（OOF 拟合，避免在 test 上挑阈值）。

测试成本：1 + K 次骨干前向（K=2 时 3 次），属于机制需要，不是 TTA。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- 区域提议
def propose_regions(cam: torch.Tensor, alpha: float = 0.2):
    """cam [B,h,w] -> 归一化框 [B,4]（x0,y0,x1,y1）。"""
    from model.cam_aug import mask_to_box, peak_mask
    return mask_to_box(peak_mask(cam, alpha).detach())


# ---------------------------------------------------------------- 路由特征
def routing_features(main_logits, region_logits, main_feat, region_feats):
    """构造逐样本的路由特征。

    main_logits   [B,C]
    region_logits [B,K,C]
    main_feat     [B,D]
    region_feats  [B,K,D]

    每个样本的特征向量由三部分组成：
      * 主分支的三个量：max prob、margin(top1-top2)、熵；
      * 每个区域的四个量：max prob、margin、与主分支的 KL、与主分支特征的余弦；
      * （不显式放 agreement 标志，KL≈0 时它已经退化为「一致」）
    """
    pm = main_logits.float().softmax(-1)
    pr = region_logits.float().softmax(-1)
    b, k, _ = pr.shape

    def stats(p):
        top2 = p.topk(2, dim=-1).values
        ent = -(p.clamp_min(1e-9).log() * p).sum(-1)
        return torch.stack([p.max(-1).values, top2[..., 0] - top2[..., 1], ent], dim=-1)

    sm = stats(pm)                                       # [B,3]
    pm_log = F.log_softmax(main_logits.float(), -1).unsqueeze(1)   # [B,1,C]
    pr_log = F.log_softmax(region_logits.float(), -1)
    kl = (pm_log.exp() * (pm_log - pr_log)).sum(-1)      # [B,K]
    cos = F.cosine_similarity(region_feats.float(),
                              main_feat.float()[:, None, :].expand_as(region_feats.float()), dim=-1)
    sr = torch.stack([pr.max(-1).values,
                      pr.topk(2, dim=-1).values.diff(dim=-1).squeeze(-1),
                      kl, cos], dim=-1)                  # [B,K,4]
    return torch.cat([sm, sr.reshape(b, k * 4)], dim=-1)


class RegionRouter(nn.Module):
    """逐样本软路由：输出 K+1 个权重（最后一个是「都不用」）。零初始化。"""

    def __init__(self, in_dim: int, num_regions: int, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.num_regions = int(num_regions)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, self.num_regions + 1),
        )
        # 零初始化 + 一个可学习的整体缩放：初始时 fused = main（权重全 0）
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        # ⚠️ 不能用 0：sigmoid(0)=0.5，初始就叠加了一半区域项（实测 |fused-main|≈0.56）。
        # 取 -10 => sigmoid≈4.5e-5，初始 fused 与 main 的差在 1e-3 量级。
        self.gamma = nn.Parameter(torch.full((1,), -10.0))

    def forward(self, feats: torch.Tensor):
        w = self.mlp(feats).softmax(dim=-1)              # [B,K+1]
        return w

    def fuse(self, main_logits, region_logits, feats):
        """fused = main + gamma * Σ_k w_k · region_logits_k（w 不带「都不用」那一项）。"""
        w = self.forward(feats)
        g = torch.sigmoid(self.gamma)
        mixed = (w[:, : self.num_regions].unsqueeze(-1) * region_logits).sum(1)
        return main_logits + g * mixed, w
