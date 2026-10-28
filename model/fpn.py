"""FPN 式多尺度融合：layer3（28×28）+ layer4（14×14）。

为什么只融两层：ResNet-50 的 layer2 是 56×56，语义太浅（GAP 后单独线性探针只有
约 49%，layer4 是 86.6%），再加一层只会稀释。FPN 的标准做法是「深层的语义上采样后
和浅层的高分辨率相加」，两层足够演示这个机制。

⚠️ 融合完立刻 GAP 的话，28×28 的分辨率优势就没了 —— 剩下的只是「融合了不同层次的
   全局统计」。要真正吃到高分辨率，得让融合后的特征去驱动 CAM / 注意力 / 局部池化。

⚠️ 本模块必须挂在 `Classifier.fpn` 上，并且 `engine/build.py` 的 build_optimizer
   要把它单独列成一组（它既不在 backbone 也不在 head 里）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNFusion(nn.Module):
    """layer3 + layer4 -> 一个 C 通道、28×28 的特征图。

    out_channels=None 时对齐到 layer4 的通道数（不降维），这样下游 head 的输入维度
    和原来完全一样，可以把变量隔离到「融不融合」本身，而不是「降维损失了多少」。
    """

    def __init__(self, c3: int, c4: int, out_channels: int | None = None,
                 smooth: bool = True):
        super().__init__()
        c = int(out_channels or c4)
        self.lat3 = nn.Conv2d(c3, c, 1)
        self.lat4 = nn.Conv2d(c4, c, 1)
        # 3×3 平滑上采样带来的棋盘效应（FPN 的标准做法，比直接用 bilinear 干净）
        self.smooth = nn.Conv2d(c, c, 3, padding=1) if smooth else nn.Identity()
        self.norm = nn.BatchNorm2d(c)
        self.out_channels = c

    def forward(self, feat3: torch.Tensor, feat4: torch.Tensor) -> torch.Tensor:
        f3 = self.lat3(feat3)                       # [B, C, 28, 28]
        f4 = self.lat4(feat4)                       # [B, C, 14, 14]
        if f4.shape[-2:] != f3.shape[-2:]:
            f4 = F.interpolate(f4, size=f3.shape[-2:], mode="nearest")
        return self.norm(self.smooth(f3 + f4))      # [B, C, 28, 28]


def build_fpn(backbone, mcfg: dict) -> FPNFusion | None:
    """按 config 建 FPN；MODEL.fpn 为 false 时返回 None（行为和以前完全一致）。"""
    if not mcfg.get("fpn", False):
        return None
    info = backbone.feature_info                     # list[dict{num_chs,reduction,module}]
    c3, c4 = info[-2]["num_chs"], info[-1]["num_chs"]
    return FPNFusion(
        c3=c3, c4=c4,
        out_channels=mcfg.get("fpn_channels"),      # None = 不降维
    )
