"""把一层全局自注意力插进骨干内部（layer3 之后、layer4 之前）。

为什么不是接在 layer4 后面：那条路已经试过四次（attn_pool 196/d256 88.18、
d512/4 层 88.51、49 token 88.82，无注意力控制臂 88.83），全部低于基线 89.40，
且控制臂说明多出来的残差路径本身就要付 0.2 的代价。原因是我们自己的探针给的：
layer4 每个格子（14×14、448 输入）的感受野已经接近整张图，特征在空间上已经
被抹平，再在这张图上重新分配权重，能翻转的信息很有限。

layer3 的输出是 28×28，和 ViT-B 的 patch 数一样，每个位置只覆盖局部。
在这里做一次全局混合，然后让**预训练的 layer4** 去消费混合后的特征 ——
信息是在进入深层之前被重组的，而不是在最后被重新加权。

设计约束与 AttnPool 相同：输出投影零初始化，训练第 0 步与基线逐位等价，
之后任何变化都只能来自这个模块本身。

数据流（448 输入，layer3 输出 [B,1024,28,28]）：
    x [B,1024,28,28]
      -> 1x1 卷积降到 dim        [B,dim,28,28]
      -> flatten + 转置          [B,784,dim]
      -> 加二维正弦位置编码
      -> N 个 pre-LN Transformer block
      -> 回到特征图              [B,dim,28,28]
      -> 1x1 卷积回 1024（零初始化）
    layer4 的输入 = x + 上式

参数量（dim=256, 1 层）：约 1.0M，相对 24M 的 R50 是 4%。
5400 张训练图容不下更大的模块——这是前面几次失败给的教训。

用法（见 model/backbone.py:build_global_context）：
    gc = GlobalContext(channels=1024, dim=256, depth=1, heads=4)
    backbone.layer4.register_forward_pre_hook(gc.pre_hook)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _sincos_2d(h: int, w: int, dim: int) -> torch.Tensor:
    """二维正弦位置编码，返回 [1, h*w, dim]。ViT 的 1D 版本按行列各算一半。"""
    assert dim % 4 == 0, "dim 必须是 4 的倍数（行/列各占一半，正弦/余弦各占一半）"
    half = dim // 2
    omega = torch.arange(half // 2, dtype=torch.float32) / (half / 2.0)
    omega = 1.0 / (10000 ** omega)
    gy = torch.arange(h, dtype=torch.float32)
    gx = torch.arange(w, dtype=torch.float32)
    out_y = torch.einsum("h,d->hd", gy, omega)
    out_x = torch.einsum("w,d->wd", gx, omega)
    out_y = torch.cat([out_y.sin(), out_y.cos()], dim=1)          # [h, half]
    out_x = torch.cat([out_x.sin(), out_x.cos()], dim=1)          # [w, half]
    pos = torch.cat([
        out_y[:, None, :].expand(h, w, half),
        out_x[None, :, :].expand(h, w, half),
    ], dim=2)                                                     # [h,w,dim]
    return pos.reshape(1, h * w, dim)


class GlobalContext(nn.Module):
    """layer3 输出上的全局自注意力，残差回原特征图，零初始化。"""

    def __init__(self, channels: int = 1024, dim: int = 256, depth: int = 1,
                 heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.0,
                 use_pos: bool = True):
        super().__init__()
        self.dim = int(dim)
        self.use_pos = bool(use_pos)
        self._pos_cache: torch.Tensor | None = None
        self.proj = nn.Conv2d(channels, dim, kernel_size=1)
        if depth > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        else:
            # 控制臂：**参数对齐**的逐点 MLP，不做 token 之间的任何混合。
            # 用 Identity 当控制臂是错的：那样只有 526K 参数（注意力臂 1.05M），
            # 一旦涨点就分不清是「全局混合」还是「多出来的那 526K」。
            # 这里 256 -> 1024 -> 256 的逐点 MLP 正好约 525K，见 PLAN_R50_90.md 的对照设计。
            self.blocks = nn.Sequential(
                nn.Linear(dim, int(dim * mlp_ratio * 2)),
                nn.GELU(),
                nn.Linear(int(dim * mlp_ratio * 2), dim),
            )
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Conv2d(dim, channels, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # 给日志用：模块输出的幅度，以及它相对 GAP 特征的比例。
        # 没有这两个量，「模块没长起来」和「机制无效」在曲线上分不开。
        self._last_out_norm = 0.0
        self._last_ratio = 0.0

    def _pos(self, h: int, w: int, device, dtype) -> torch.Tensor:
        if self._pos_cache is None or self._pos_cache.shape[1] != h * w \
                or self._pos_cache.shape[2] != self.dim:
            self._pos_cache = _sincos_2d(h, w, self.dim)
        return self._pos_cache.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        t = self.proj(x).flatten(2).transpose(1, 2)               # [B,N,dim]
        if self.use_pos:
            t = t + self._pos(h, w, t.device, t.dtype)
        t = self.blocks(t)
        t = self.norm(t)
        t = t.transpose(1, 2).reshape(b, self.dim, h, w)
        y = self.out(t)
        with torch.no_grad():
            # 逐位置取通道向量的范数再对位置平均：衡量「加进去的东西」有多大，
            # 和它相对原特征的比例。y 初始恒为 0，所以这两个量从 0 开始长。
            self._last_out_norm = float(y.detach().flatten(2).norm(dim=2).mean())
            g = x.detach().flatten(2).norm(dim=2).mean()
            self._last_ratio = float(
                self._last_out_norm / g.clamp(min=1e-12))
        return x + y

    def pre_hook(self, module, inputs):
        """给 layer4 注册的 forward_pre_hook：把输入换成混合后的特征。"""
        return (self.forward(inputs[0]),) + tuple(inputs[1:])


def build_global_context(backbone: nn.Module, mcfg: dict) -> GlobalContext | None:
    """按 config 在 layer4 前插一层 GlobalContext；未开启时返回 None。"""
    if not mcfg.get("global_ctx", False):
        return None
    if not hasattr(backbone, "layer4"):
        raise ValueError("MODEL.global_ctx 目前只支持 ResNet 系列骨干")
    if getattr(backbone, "global_ctx", None) is not None:
        # 重复调用会再挂一个 hook，两个模块叠加，结果不可解释
        raise RuntimeError("global_ctx 已经注册过了，不要重复调用 build_global_context")
    # layer3 输出通道：ResNet-50 是 1024。从模型自身读，不写死。
    ref = backbone.layer3[-1]
    ch = getattr(getattr(ref, "conv3", None), "out_channels", None)
    if ch is None:
        raise ValueError("layer3 的最后一个 block 没有 conv3（不是 Bottleneck），"
                         "请改用 feature_info 里的通道数")
    gc = GlobalContext(
        channels=ch,
        dim=int(mcfg.get("gc_dim", 256)),
        depth=int(mcfg.get("gc_depth", 1)),
        heads=int(mcfg.get("gc_heads", 4)),
        mlp_ratio=float(mcfg.get("gc_mlp", 2.0)),
        dropout=float(mcfg.get("gc_dropout", 0.0)),
        use_pos=bool(mcfg.get("gc_pos", True)),
    )
    # 跟随 backbone 当前的 device/dtype：如果有人在 .to(cuda) 之后才调用，
    # 不搬过去的话第一次 forward 会报 dtype 不匹配
    try:
        p = next(backbone.parameters())
        gc.to(device=p.device, dtype=p.dtype)
    except StopIteration:
        pass
    backbone.add_module("global_ctx", gc)
    backbone.layer4.register_forward_pre_hook(gc.pre_hook)
    return gc
