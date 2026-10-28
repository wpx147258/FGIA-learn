"""CAP: Context-aware Attentional Pooling (Behera et al., AAAI 2021).

计算图（arXiv:2101.06635 / github.com/ArdhenduBehera/cap）：

    backbone 特征图 [B,C,h,w]
      → SE                      通道重标定
      → 像素级自注意力            non-local / SAGAN，每个像素条件于全图
      → 上采样到 42×42           给 ROI 池化留出亚像素精度
      → 27 个几何 ROI            3×3 anchor（0/14/28/42），边长≥28 的矩形
      → 各 ROI 池到 7×7 → GAP    roi_align
      → 区域间加性注意力          每个区域条件于所有其他区域（论文 Eq.2）
      → 27 个区域取平均           论文的「标准分类器」（B+C）
      → FC → logits

默认是论文的 **B+C**（+11.8）。cap_vlad=True 再加 LSTM+NetVLAD（B+C+E，再 +4.4）。

## 和官方实现的差异（都是为了不退化，不是等价的等价变换）

官方把每个 ROI 拉平成 7×7×C = 100,352 维，直接在上面做区域注意力，于是
`Wβ/Wβ'` 各 3.2M，加上像素自注意力的 h 卷积 2048×2048 = 4.2M，头部共 13.4M。

  1. **区域注意力作用在 GAP 后的 C 维**（light=True），参数 6.4M → 0.13M。
     注意这是**有损**的：官方版本能看到 ROI 内 7×7 各位置的分布，GAP 之后
     只能看到均值。表达力是真子集，不是等价。
  2. **像素自注意力的 value 用原特征 x**，省掉 h 卷积（SAGAN 的写法）。
  3. **区域注意力保持论文的加性形式**。实测它在第 9 轮时 93.4% 的条目 tanh
     饱和、平均导数 0.025，看起来像「死掉了」，于是换成了 LN + 点积 —— 结果
     3 轮内塌成 one-hot（logits std 44.8，熵 0%，alpha_max 1.000），val 从
     50.8 掉到 21.1。tanh 的饱和不是 bug，是唯一在阻止注意力塌缩的刹车。
     代价是注意力只能停在接近均匀的位置（不做选择），但这个功能并没有被证明
     对最终精度有贡献。
  4. **γ 从 0.1 起步**，不从 0。非局部块的输出幅度只有特征的 ~1/√N，γ=0 时
     f/g 收不到梯度（∂L/∂W ∝ γ），实测 10 轮后 γ 仍是 -0.0005，整块没接上。
  5. **NetVLAD 描述子乘回 √dim**。L2 归一化后逐维方差中位数 ~4e-8，比
     BatchNorm 的 eps(1e-5) 小两个数量级，√(var+eps) 被 eps 主导，BN 的方差
     归一化失效（实测输出 rms 0.144，应 ~1）。

light=False 可以退回官方的重参数版本（3/4/5 仍然生效）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


def make_rois(resolution: int = 42, grid: int = 3, min_size: int = 28):
    """几何 ROI：3×3 anchor，保留「宽 ≥ min_size 或 高 ≥ min_size」的矩形。

    官方 getROIS(resolution=42, gridSize=3, minSize=2) 得到 26 个 + 整图 = 27。
    这里直接返回全部 27 个（含整图）。
    """
    step = resolution // grid
    anchors = [i * step for i in range(grid + 1)]
    boxes = []
    for x0 in anchors:
        for x1 in anchors:
            for y0 in anchors:
                for y1 in anchors:
                    w, h = x1 - x0, y1 - y0
                    if w > 0 and h > 0 and (w >= min_size or h >= min_size):
                        boxes.append((x0, y0, x1, y1))
    return boxes


class SEBlock(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        hid = max(c // r, 8)
        self.fc1 = nn.Conv2d(c, hid, 1)
        self.fc2 = nn.Conv2d(hid, c, 1)

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        return x * torch.sigmoid(self.fc2(F.relu(self.fc1(s))))


class PixelSelfAttention(nn.Module):
    """non-local：x ← x + γ·(全图所有位置的加权和)。γ 初始 0，起点是恒等。"""

    def __init__(self, c: int, reduction: int = 16, use_h: bool = False):
        super().__init__()
        hid = max(c // reduction, 8)
        self.f = nn.Conv2d(c, hid, 1)
        self.g = nn.Conv2d(c, hid, 1)
        self.h = nn.Conv2d(c, c, 1) if use_h else None
        # γ 从 0.1 起步，不是 0。o 是对全图 196 个位置取加权平均，幅度天然只有
        # 特征本身的 ~1/√196；γ=0 时 f/g 的梯度是 γ·(...)，正好为 0，整块起不来。
        # 实测（旧版）：训 10 轮后 γ 还是 -0.0005，自注意力全程没参与。
        self.gamma = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        f = self.f(x).view(B, -1, N).permute(0, 2, 1)     # [B,N,HID]
        g = self.g(x).view(B, -1, N)                      # [B,HID,N]
        beta = torch.softmax(torch.bmm(f, g), dim=-1)     # [B,N,N]
        v = (self.h(x) if self.h is not None else x).view(B, C, N)
        o = torch.bmm(v, beta.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * o + x


class CAPHead(nn.Module):
    on_feature_map = True     # 需要池化前的特征图，Classifier 自动改路径

    def __init__(self, in_features: int, out_features: int,
                 resolution: int = 42, pool_size: int = 7,
                 attn_units: int = 32, attn_reduction: int = 16,
                 dropout: float = 0.0, use_vlad: bool = False,
                 lstm_hidden: int = 128, clusters: int = 32,
                 light: bool = True, region: str = "attn"):
        super().__init__()
        C = in_features
        self.se = SEBlock(C)
        self.pixel_attn = PixelSelfAttention(
            C, reduction=attn_reduction, use_h=not light)

        self.resolution = resolution
        self.pool_size = pool_size
        rois = make_rois(resolution)
        self.register_buffer("rois", torch.tensor(rois, dtype=torch.float32),
                             persistent=False)
        self.num_regions = len(rois)

        # 区域注意力的作用维度：light 用 GAP 后的 C 维，faithful 用 7*7*C 维
        self.light = bool(light)
        d_attn = C if light else C * pool_size * pool_size
        # 论文 Eq.2 的**加性**注意力 w^T tanh(W_i r_i + W_j r_j + b)。
        # 曾经试过换成「LayerNorm → 点积 → 1/√d」，理由是 tanh 会饱和。
        # 结果 3 轮内注意力塌成 one-hot（logits std 44.8，熵 0%×ln27，
        # alpha_max=1.000），27 个区域退化成 1 个，val 从 50.8 掉到 21.1。
        # 教训：tanh 饱和在这里不是 bug，而是唯一在阻止注意力塌缩的刹车。
        # 加性形式保留。
        # region="identity" 时不做跨区域注意力，27 个区域各自保留自己的特征。
        # 动机（实测）：α 的熵 97%×ln27 ≈ 近均匀，于是 ctx_r = Σ_j α_rj R_j
        # 对每个 r 都等于 R 的均值 —— 27 行几乎完全相同（离散度 0.0004，R 本身
        # rms 0.16）。再经过 LSTM→NetVLAD 的第二次平均，不同图片的描述子两两
        # 余弦高达 0.998，分类器只在 0.002 量级的残差上工作，这也解释了训练中
        # 的大幅震荡。identity 是判断「27 区域结构到底有没有信息」的对照。
        self.region = region
        if region == "identity":
            self.rq = self.rk = self.ra = None
            self.rb = None
        else:
            self.rq = nn.Linear(d_attn, attn_units)
            self.rk = nn.Linear(d_attn, attn_units)
            self.ra = nn.Linear(attn_units, 1)
            self.rb = nn.Parameter(torch.zeros(attn_units))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.use_vlad = bool(use_vlad)
        if self.use_vlad:
            self.lstm = nn.LSTM(C, lstm_hidden, batch_first=True)
            self.vlad_assign = nn.Linear(lstm_hidden, clusters)
            self.bn = nn.BatchNorm1d(clusters * lstm_hidden)
            self.fc = nn.Linear(clusters * lstm_hidden, out_features)
            self.clusters = clusters
            self.dim = clusters * lstm_hidden
        else:
            self.fc = nn.Linear(C, out_features)
            self.clusters = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.se(x)
        x = self.pixel_attn(x)

        B, C, H, W = x.shape
        m = F.interpolate(x, size=(self.resolution, self.resolution),
                          mode="bilinear", align_corners=False)

        K = self.num_regions
        idx = torch.arange(B, device=x.device).repeat_interleave(K)
        boxes = torch.cat([idx[:, None].float(),
                           self.rois.repeat(B, 1)], dim=1)      # [B*K, 5]
        roi = roi_align(m, boxes, output_size=(self.pool_size, self.pool_size),
                        spatial_scale=1.0, aligned=False)

        if self.light:
            R = roi.mean(dim=(2, 3)).view(B, K, C)              # [B,K,C]
        else:
            R = roi.flatten(1).view(B, K, -1)                   # [B,K,7*7*C]

        if self.region == "identity":
            ctx = R                                             # 每个区域保留自己
        else:
            q = self.rq(R)                                      # [B,K,32]
            k = self.rk(R)
            beta = torch.tanh(q.unsqueeze(2) + k.unsqueeze(1) + self.rb)
            alpha = torch.softmax(self.ra(beta).squeeze(-1), dim=-1)  # [B,K,K]
            ctx = torch.bmm(alpha, R)                           # [B,K,d]

        if not self.light:
            ctx = ctx.view(B, K, C, self.pool_size, self.pool_size).mean(dim=(3, 4))

        if not self.use_vlad:
            return self.fc(self.dropout(ctx.mean(dim=1)))

        h, _ = self.lstm(ctx)                                   # [B,K,L]
        a = torch.softmax(self.vlad_assign(h), dim=-1)          # [B,K,clusters]
        N = torch.einsum("brk,brl->bkl", a, h)                  # [B,clusters,L]
        # L2 归一化后整条向量范数是 1，4096 维平均每维只有 0.0156，逐维方差中位数
        # ~4e-8 —— 比 BatchNorm 默认的 eps(1e-5) 还小两个数量级。于是 √(var+eps)
        # 被 eps 主导，BN 退化成「减均值 + 固定缩小 316 倍」，方差归一化完全失效
        # （实测输出 rms 0.144，正常应 ~1）。乘回 √dim 把尺度还原成 O(1)，
        # 只乘标量，向量方向不变，NetVLAD 的几何没动。
        N = torch.nn.functional.normalize(N.flatten(1), dim=1) * self.dim ** 0.5
        return self.fc(self.dropout(self.bn(N)))
