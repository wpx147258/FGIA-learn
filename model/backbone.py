"""timm backbone 包装：只出特征，不接分类头。

分工：
  * backbone —— 「图 -> 特征向量」，由 timm 提供，换模型只改 config 的 MODEL.name
  * 分类头   —— model/head.py
  * 两者组合 —— model/classifier.py

`num_classes=0` 是 timm 的约定，表示「不建分类头，直接返回池化后的特征」。
拿到的是纯特征提取器，特征维度用 `num_features` 读。

后续做部位定位 / 局部特征（attention、bilinear pooling）时会用到 timm 原生的两个方法：
  * `backbone.forward_features(x)` -> 池化**前**的特征图 [B, C, H', W']
  * `backbone.forward_head(feat)`  -> 池化后的向量
现在用不上，但它们是现成的，不用改这个文件。
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class GeM(nn.Module):
    """广义平均池化：f = (mean(x^p))^(1/p)，p 可学习。

    p=1 就是平均池化（GAP），p→∞ 就是最大池化。中间值相当于「比平均更看重
    强响应的那些格子，但不至于像 max 那样只留一个」。检测/检索里常用，
    作为 GAP 的替身只多一个标量参数，接口与 timm 的 global_pool(flatten=True)
    一致：输入 [B,C,H,W]，输出 [B,C]。
    在 fp32 下算，避免 bf16 里 pow 的数值问题。
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().clamp(min=self.eps)
        return x.pow(self.p).mean(dim=(2, 3)).pow(1.0 / self.p)


class SE(nn.Module):
    """通道注意力（Squeeze-and-Excitation）：全局池化出通道描述子，
    经两个 1×1 卷积学到每个通道的权重，再乘回原特征图。
    插入到 ResNet 每个 bottleneck 之后，参数约 2.8M。"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(8, channels // reduction)
        self.reduce = nn.Conv2d(channels, mid, 1)
        self.expand = nn.Conv2d(mid, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        s = torch.sigmoid(self.expand(torch.relu(self.reduce(s))))
        return x * s


def add_se_blocks(model: nn.Module, reduction: int = 16) -> nn.Module:
    """把 SE 插到每个 Bottleneck 的**残差分支**上（标准 SE-ResNet）。

    标准 SE-ResNet（Hu et al. 2018）：out = relu(x + SE(F(x)))。
    SE 只包残差分支 F(x)（conv1..bn3），捷径 x 原样通过。

    实现：在 bn3（残差分支的最后一层）上挂 forward hook，把它的输出套上 SE。
    hook 返回值会替换 bn3 的输出，下游的加法用的就是 SE(F(x))，捷径不受影响。

    为什么用 hook 而不是替换整个 forward：
      旧版用 lambda 把 SE 包在整个 block 输出（SE(x+F(x))）上，捷径被一起衰减，
      16 个 block 连乘 ≈0.5^16，特征直接归零（实测 epoch 0 只有 1% 准确率）。
      hook 精确插在「bn3 之后、加法之前」，等价于标准 SE-ResNet 的位置。
    """
    n = 0
    for m in model.modules():
        if type(m).__name__ == "Bottleneck":
            c = m.conv3.out_channels
            se = SE(c, reduction)
            # 独立命名 se_block，避免与 timm 原生的 self.se 槽位冲突（否则双重应用）
            m.se_block = se

            def _hook(module, args, output, _se=se):
                return _se(output)   # SE(F(x))

            m.bn3.register_forward_hook(_hook)
            n += 1
    if n == 0:
        raise ValueError("没找到 Bottleneck，SE 插入失败")
    return model


def set_output_stride8(model: nn.Module) -> nn.Module:
    """把 ResNet 的 layer4 下采样关掉（dilated），特征图分辨率翻倍。

    448 输入下最终特征图从 14×14 变成 28×28 —— 和 ViT-B 的 patch 数一样，
    是「R50 是不是被最后一组下采样抹掉了细节」这个假设的直接检验。

    具体改动（timm 的 Bottleneck）：
      * 每个 block 的 conv2：stride (2,2) -> (1,1)，dilation -> (2,2)，padding -> (2,2)
        （3×3 + dilation d 必须 padding=d 才保持尺寸，padding=1 会让 28->26，
          和捷径分支对不上）；
      * 第一个 block 的 downsample 1×1 conv：stride (2,2) -> (1,1)。
    权重张量形状全都不变，预训练权重可以原样加载。

    ⚠️ 两处与「标准做法」不同，记录在案：
      1) 名字里的 stride8 是 224 输入下的叫法。448 输入时 layer3 输出已经是 28×28，
         所以这里的输出 stride 实际是 16，不是 8。不要和 timm 的 output_stride=8
         混用（那个连 layer3 一起 dilation，会得到 56×56）。
      2) torchvision/timm 的 replace_stride_with_dilation 会保留第一个 block 的
         dilation=1（只把后续 block 设成 2）；这里第一个 block 也设成 dilation=2。
         两者 layer4 输出的相对差异约 1.4（mean|Δ|/mean|f|），不是数值细节。
         第一版实验（r50_s8）用的就是这里这一版；若要标准版，需另加开关重跑。
    """
    if not hasattr(model, "layer4"):
        raise ValueError(f"{type(model).__name__} 没有 layer4，stride8 目前只支持 ResNet")
    blocks = list(model.layer4.children())
    for blk in blocks:
        if not hasattr(blk, "conv2"):
            raise ValueError(f"layer4 里有非 Bottleneck 模块: {type(blk).__name__}")
        blk.conv2.stride = (1, 1)
        blk.conv2.dilation = (2, 2)
        blk.conv2.padding = (2, 2)
    down = getattr(blocks[0], "downsample", None)
    if down is not None and hasattr(down[0], "stride"):
        down[0].stride = (1, 1)
    # feature_info 在 timm 里是普通 list（不是带 .info 的对象）。写错时这里会
    # 静默失败，reduction 一直是 32；没有代码消费它，但诊断脚本可能会。
    try:
        model.feature_info[-1]["reduction"] = 16
    except Exception:
        pass
    return model


def build_backbone(name: str, pretrained: bool = True, pool: str = "avg",
                   gem_p: float = 3.0, se: bool = False, se_reduction: int = 16,
                   stride8: bool = False, content_pool: bool = False,
                   content_rank: int = 64,
                   **kwargs) -> torch.nn.Module:
    """kwargs 透传给 timm.create_model，用来开 backbone 自身的正则。

    目前用得上的只有 drop_path_rate（随机深度，a1 预训练用的是 0.05）。
    pool="gem" 时把 timm 的 global_pool 换成可学习的广义平均池化。
    stride8=True 时把 layer4 改成 dilated（输出 stride 8），见 set_output_stride8。
    content_pool=True 时把全局池化换成 GAP 初始化的内容注意力池化（见 attn_pool.py）。
    """
    model = timm.create_model(name, pretrained=pretrained, num_classes=0, **kwargs)
    if pool and pool != "avg":
        if pool != "gem":
            raise KeyError(f"未知的池化方式: {pool!r}，可选 avg | gem")
        if not hasattr(model, "global_pool"):
            raise ValueError(f"{name} 没有 global_pool 属性，不能替换池化")
        model.global_pool = GeM(p=gem_p)
    if se:
        add_se_blocks(model, se_reduction)
    if stride8:
        set_output_stride8(model)
    if content_pool:
        if pool and pool != "avg":
            raise ValueError("MODEL.content_pool 不能和 MODEL.pool=gem 同时开启")
        from model.attn_pool import ContentPool
        model.global_pool = ContentPool(int(model.num_features), int(content_rank))
    return model


if __name__ == "__main__":
    backbone = build_backbone("resnet50", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    print("num_features    :", backbone.num_features)
    print("forward(x)      :", tuple(backbone(x).shape))                     # [2, 2048]
    print("forward_features:", tuple(backbone.forward_features(x).shape))    # [2, 2048, 7, 7]
