"""CNN 特征图 -> token 序列 -> 轻量自注意力（综述 V-D 指向的那类「部位间相互依赖」）。

动机：
  综述里那段原话是 "jointly learn the interdependencies among part-level features"。
  我们此前所有部位方法（三流、CAP、NetVLAD、CAM 裁剪）都是「把某处取出来再用」，
  没有一个是让部位之间互相看。这个模块做的就是让 49 个位置互相看。

设计上有一条硬约束：**初始时必须等于基线**。
  输出层零初始化，所以训练第 0 步时模块贡献为 0，模型和 89.40 那份完全等价。
  这样训练后的任何变化都只能来自这个模块本身，而不是"多了一层初始化扰动"。

数据流（以 448 输入、layer4 输出 [B,2048,14,14] 为例）：
    feat [B,2048,14,14]
      -> 2x2 平均池化      [B,2048,7,7]      （tokens=49 时）
      -> 1x1 卷积降维      [B,dim,7,7]
      -> flatten + 转置    [B,49,dim]
      -> 加可学习位置编码
      -> N 个 Transformer block
      -> token 平均         [B,dim]
      -> 线性层回到 2048    [B,2048]   （零初始化）
    最终特征 = GAP(feat) + 上式         （残差相加，不是拼接）

为什么用相加而不是拼接：拼接会改变分类头输入维度，等于换了一个头；
相加保持 2048 维，头完全不动，基线的参数量与结构不变。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttnPool(nn.Module):
    """把特征图压成 token 序列做自注意力，输出与池化特征同维的残差。"""
    ### Q ：采用 readout: = "cls" 是否建议depth>1?

    def __init__(self, in_channels: int = 2048, out_dim: int = 2048, dim: int = 768,
                 tokens: int = 196, depth: int = 1, heads: int = 8,
                 mlp_ratio: float = 2.0, dropout: float = 0.0, use_pos: bool = False,
                 readout: str = "mean"):
        super().__init__()
        self.tokens = int(tokens)
        self.use_pos = bool(use_pos)
        if readout not in ("mean", "cls"):
            raise ValueError(f"readout 只能是 mean | cls，收到 {readout!r}")
        self.readout = readout
        # 49 = 先在特征图上做 2x2 平均池化；其余（196 / 784）= 不池化，直接用全部位置。
        # 文献调研的结论：细粒度上主流是 1x1 降维后 flatten 全部位置，
        # 没有工作用 2x2 池化到 49，所以 49 只作为最后面对照保留。
        self.pre_pool = nn.AvgPool2d(2) if tokens == 49 else nn.Identity()
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=1)
        if self.use_pos:
            self.pos = nn.Parameter(torch.zeros(1, tokens, dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
        else:
            self.pos = None

        if depth > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        else:
            # depth=0 是控制臂：不接注意力，只保留「1x1 投影 -> token 平均 -> 线性」。
            # 用来分离「多出来的参数与池化」和「注意力本身」的贡献。
            self.blocks = nn.Identity()
        self.norm = nn.LayerNorm(dim)

        # 零初始化：模块初始输出恒为 0，模型与基线逐位等价
        self.out = nn.Linear(dim, out_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat [B,C,H,W] -> [B,C]，与 GAP 输出同维，供直接相加。"""
        x = self.pre_pool(feat)
        x = self.proj(x).flatten(2).transpose(1, 2)     # [B,N,dim]
        if self.readout == "cls":
            # 全局 token：初始 = N 个 token 的平均（GAP），拼在序列最前面。
            # 它不带位置编码（是全局表示，不是某个空间格子）；经过自注意力后
            # 吸收其它 token 的信息，读出它自己作为聚合结果。
            gap = x.mean(dim=1)                          # [B,dim]
            if self.pos is not None:
                x = x + self.pos[:, : x.size(1)]
            t = torch.cat([gap.unsqueeze(1), x], dim=1)  # [B,N+1,dim]
            t = self.blocks(t)
            x = self.norm(t)[:, 0]                       # 读第 50 个 token（更新后）
        else:
            if self.pos is not None:
                x = x + self.pos[:, : x.size(1)]
            x = self.blocks(x)
            x = self.norm(x).mean(dim=1)                 # token 平均
        y = self.out(x)
        # 记两个量给日志：模块输出的绝对范数，以及它与 GAP 特征的相对幅度。
        # 后者才是关键 —— 残差要从 0 长到和 GAP 可比的数量级，才可能真正影响分类。
        # 训练和推理都记：否则指标要等第一次验证才出现，CSV 表头会缺列
        with torch.no_grad():
            g = feat.mean(dim=(2, 3))
            self._last_out_norm = float(y.detach().norm(dim=1).mean())
            self._last_ratio = float(
                (y.detach().norm(dim=1).mean() / g.norm(dim=1).mean().clamp(min=1e-12)))
        return y


class SEGate(nn.Module):
    """残差式通道门控：out = feat * (1 + delta)，delta 初始 0。

    旧版是绝对门控（out = feat * sigmoid(...)），乘在完整特征上、没有捷径保底，
    门学偏会整体缩放特征。这里改成**乘性残差**：缩放因子 = 1 + delta，delta 从 0
    学起，初始恒等（等价基线），学到的是「相对恒等的缩放增量」。
    delta 用 tanh 限制在 (-1,1)，保证 1+delta ∈ (0,2) 恒为正，通道不会被翻转为负。

    为什么加在最终特征图上而不是插进每个 bottleneck：标准 SE-ResNet（残差分支上）
    现在由 MODEL.se / backbone.add_se_blocks 提供；这里只做最终特征图上的一次
    乘性残差门控，作为更轻的对照。
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(8, channels // reduction)
        self.reduce = nn.Conv2d(channels, mid, 1)
        self.expand = nn.Conv2d(mid, channels, 1)
        # delta 初始 0：expand 零初始化 -> tanh(0)=0 -> 1+delta=1（恒等）
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        s = feat.mean(dim=(2, 3), keepdim=True)
        delta = torch.tanh(self.expand(torch.relu(self.reduce(s))))
        return feat * (1.0 + delta)


class TokenLearner(nn.Module):
    """软空间注意力池化（TokenLearner, NeurIPS 2021）。

    与 AttnPool 的区别：AttnPool 让 196 个 token 互相看，输出仍是"整体的一个向量"；
    这里则是学 S 张空间权重图，每张图把 196 个位置加权求和成一个 token，
    得到 S 个"部位级"表示再平均。它不改 token 之间的关系，只改**聚合方式**。

    为什么值得试：我们的诊断显示
      - 61.9% 的 token 对真实类贡献为负，但硬剪掉它们不改变准确率（负贡献量级太小）
      - 贡献最高的 20% token 占正向质量 96.9%
    硬剪无效，但"按可学习的空间权重重新加权"未必无效 —— 这是两者唯一没被排除的差别。

    输出层零初始化，训练第 0 步与基线等价；同时返回 S 张权重图的熵，
    熵接近 ln(196) 说明权重退化成均匀（等于 GAP），接近 0 说明塌到单个位置。
    """

    def __init__(self, in_channels: int = 2048, out_dim: int = 2048, dim: int = 256,
                 s_tokens: int = 8, dropout: float = 0.0,
                 ent_target: float = 0.0, div_w: float = 0.0):
        super().__init__()
        self.s = int(s_tokens)
        self.dim = int(dim)
        self.proj = nn.Conv2d(in_channels, dim, 1)
        self.score = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, self.s),
        )
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, out_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self._last_entropy = None
        # 防塌缩：不加约束时实测 8 张权重图的熵会掉到 0.004（全部塌成单点），
        # 等于退化成"学出来的 top-8 池化"。两个正则：
        #   ent_target：把每张图的熵往目标值拉（均匀=5.28，单点=0，目标取 2 左右）
        #   div_w：惩罚 8 张图之间的重叠，逼它们覆盖不同区域
        self.ent_target = float(ent_target)
        self.div_w = float(div_w)
        self._reg = None

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.proj(feat).flatten(2).transpose(1, 2)      # [B,N,dim]
        logits = self.score(x)                              # [B,N,S]
        w = logits.softmax(dim=1)                           # 对位置归一化
        z = torch.einsum("bns,bnd->bsd", w, x)              # [B,S,dim]  每张图一个 token
        v = self.norm(z).mean(dim=1)
        ent = -(w * w.clamp_min(1e-9).log()).sum(dim=1)      # [B,S] 每张图一个熵
        with torch.no_grad():
            self._last_entropy = float(ent.mean())
        reg = None
        if self.ent_target > 0:
            reg = ((ent - self.ent_target) ** 2).mean()
        if self.div_w > 0:
            # w 在位置上归一化，所以两张图的点积就是重叠度：0 = 不重叠，1 = 完全重合
            ov = torch.einsum("bns,bms->bsm", w, w)
            s_ = ov.size(1)
            off = (ov.sum(dim=(1, 2)) - ov.diagonal(dim1=1, dim2=2).sum(1)) / (s_ * (s_ - 1))
            dv = off.mean()
            reg = dv if reg is None else reg + dv
        self._reg = reg
        return self.out(v)


class ContentPool(nn.Module):
    """内容相关的注意力池化，**初始化时逐位等于 GAP**。

    与 AttnPool / TokenLearner 的关键差别在初始化方式：
      * 前两者把输出层零初始化（初始等价于基线，但注意力权重在第 0 步拿不到梯度，
        必须先等输出层长出来，才能反过来塑造注意力）——这是一次冷启动；
      * 这里把「均匀注意力」直接写成初始化：q 与 Wk 全零 => logits 全零 =>
        softmax 均匀 => attn 加权和恒等于 GAP。于是初始输出与 GAP 逐位相等，
        但梯度从第 0 步就能流进 q 和 Wk。

    为什么这么改：CLIP-RN50 的冻结特征探针给了一个很直接的数字 ——
    同一套权重，GAP 头 37.73%、注意力池化头 59.93%（CUB val）。注意力池化不是
    "可有可无的加料"，但它是在预训练阶段长出来的。我们要在 5400 张图上从零学，
    唯一能改善的就是别让它从"零输出"开始学。

    结构（value 直接用特征，不做输出投影，参数量只有 C×r + r）：
        k    = X Wk                     [B,N,r]
        attn = softmax(q·k / sqrt(r))   [B,N]
        pool = attn @ X                 [B,C]   （均匀时 == GAP）
        out  = a * pool + (1-a) * GAP   a = sigmoid(alpha)，初始 0.5
    """

    def __init__(self, channels: int, rank: int = 64):
        super().__init__()
        self.channels = int(channels)
        self.rank = int(rank)
        self.wk = nn.Linear(channels, rank, bias=False)
        self.q = nn.Parameter(torch.zeros(rank))
        self.alpha = nn.Parameter(torch.zeros(1))     # sigmoid(0)=0.5
        nn.init.zeros_(self.wk.weight)
        self._last_entropy = 0.0

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        x = feat.flatten(2).transpose(1, 2)                  # [B,N,C]
        k = self.wk(x)                                       # [B,N,r]
        logits = (k * self.q).sum(-1) / (self.rank ** 0.5)   # [B,N]
        attn = logits.softmax(dim=-1)
        pool = (attn.unsqueeze(-1) * x).sum(dim=1)           # [B,C]
        gap = x.mean(dim=1)
        a = torch.sigmoid(self.alpha)
        with torch.no_grad():
            ent = -(attn * attn.clamp_min(1e-9).log()).sum(1).mean()
            self._last_entropy = float(ent)
        return a * pool + (1.0 - a) * gap
