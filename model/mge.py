"""多粒度专家头（MGE-CNN 那一族的简化版，不引入新的前向）。

在同一张图上挂 4 个分类头，特征来自不同层级/不同池化：
  layer2 GAP (512)、layer3 GAP (1024)、layer4 GAP (2048)、layer4 GMP (2048)
它们的错误相关性天然低于「同一张 layer4 特征图上的四个部件」——这是文献里
「结构异质」的具体落实（Abe 2023 / Jeffares 2023 反对联合 diversity 损失，
但支持结构/训练异质）。

测试仍是单次前向：主分支出答案，4 个专家可以当否决票（离线先验证）。

⚠️ 每个专家头都带 CE，会通过梯度影响骨干——这是与「冻结骨干」不同的取舍，
   需要在结果里单独说明。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MGEHeads(nn.Module):
    def __init__(self, dims=(512, 1024, 2048), num_classes: int = 200,
                 use_max_last: bool = True, dropout: float = 0.0):
        super().__init__()
        self.use_max_last = bool(use_max_last)
        self.heads = nn.ModuleList([nn.Linear(d, num_classes) for d in dims])
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._feats: dict[str, torch.Tensor] = {}

    def hook(self, name):
        def _h(module, inputs, output):
            self._feats[name] = output
            return None
        return _h

    def forward(self):
        outs = []
        for name, head in zip(("l2", "l3", "l4"), self.heads[:3]):
            f = self._feats.get(name)
            if f is None:
                return None
            outs.append(head(self.drop(f.float().mean(dim=(2, 3)))))
        if self.use_max_last:
            f = self._feats.get("l4")
            if f is not None:
                outs.append(self.heads[3](self.drop(f.float().amax(dim=(2, 3)))))
        return torch.stack(outs, dim=1)          # [B,K,num_classes]


def build_mge(backbone: nn.Module, mcfg: dict, num_classes: int):
    if not mcfg.get("mge", False):
        return None
    if not hasattr(backbone, "layer4"):
        raise ValueError("MODEL.mge 目前只支持 ResNet")
    dims = (backbone.layer2[-1].conv3.out_channels,
            backbone.layer3[-1].conv3.out_channels,
            backbone.layer4[-1].conv3.out_channels,
            backbone.layer4[-1].conv3.out_channels)
    mge = MGEHeads(dims=dims, num_classes=int(num_classes),
                   use_max_last=bool(mcfg.get("mge_max", True)),
                   dropout=float(mcfg.get("mge_dropout", 0.0)))
    for layer, key in (("layer2", "l2"), ("layer3", "l3"), ("layer4", "l4")):
        getattr(backbone, layer).register_forward_hook(mge.hook(key))
    return mge
