"""PMG：渐进多粒度拼图训练（Du et al., ECCV 2020）。

官方实现（PRIS-CV/PMG-Progressive-Multi-Granularity-Training）读下来的要点，
和我们之前的理解有两处不同，这里按官方来：

  1. "progressive" 不是按 epoch 分阶段。**每个 batch 都跑全部四个粒度**：
     n=8、n=4、n=2 的拼图，以及原图。四次 backbone 前向。
  2. 推理只用原图，走 classifier_concat。所以部署时是**单模型单前向**，
     三路 stage 损失只是在训练时对拼图输入做辅助监督，不构成集成。

网络（对应官方 model.py）：
    layer2(512) → conv_block1 → 1024 维 → classifier1     （输入：n=8 拼图）
    layer3(1024)→ conv_block2 → 1024 维 → classifier2     （输入：n=4 拼图）
    layer4(2048)→ conv_block3 → 1024 维 → classifier3     （输入：n=2 拼图）
    三路 1024 拼接 → classifier_concat                    （输入：原图）

    每个 conv_block = 1×1 Conv(→512)+BN+ReLU，再 3×3 Conv(→1024)+BN+ReLU
    每个 classifier = BN → Linear(→512) → BN → ELU → Linear(→num_classes)

损失 = CE(xc1) + CE(xc2) + CE(xc3) + 2·CE(x_concat)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, cin: int, cout: int, k: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, padding=k // 2, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


def _classifier(dim: int, feature_size: int, num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        nn.Linear(dim, feature_size),
        nn.BatchNorm1d(feature_size),
        nn.ELU(inplace=True),
        nn.Linear(feature_size, num_classes),
    )


def jigsaw(x: torch.Tensor, n: int) -> torch.Tensor:
    """把图切成 n×n 块后随机重排再拼回。逐样本独立置换。

    输入 [B,3,H,W]（已归一化），输出同形状。H、W 必须能被 n 整除，
    不能被整除时先裁到最近的整除尺寸再插值回来（448/8=56 整除，正常不会触发）。
    """
    B, C, H, W = x.shape
    if H % n or W % n:
        H2, W2 = (H // n) * n, (W // n) * n
        x = nn.functional.interpolate(x, size=(H2, W2), mode="bilinear", align_corners=False)
        H, W = H2, W2
    hs, ws = H // n, W // n
    # [B,C,n,hs,n,ws] -> [B,n*n,C,hs,ws]
    patches = x.view(B, C, n, hs, n, ws).permute(0, 2, 4, 1, 3, 5).reshape(B, n * n, C, hs, ws)
    idx = torch.stack([torch.randperm(n * n, device=x.device) for _ in range(B)])
    patches = patches.gather(1, idx.view(B, n * n, 1, 1, 1).expand(-1, -1, C, hs, ws))
    out = patches.view(B, n, n, C, hs, ws).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
    return out


class PMGHead(nn.Module):
    """三路 conv_block + 四个分类器。只负责特征图 -> logits，不含 backbone。"""

    def __init__(self, c2: int, c3: int, c4: int, num_classes: int,
                 feature_size: int = 512):
        super().__init__()
        fs, mid = feature_size, feature_size * 2      # 官方：768→1024（这里 fs=512, mid=1024）
        self.block1 = nn.Sequential(BasicConv(c2, fs, 1), BasicConv(fs, mid, 3))
        self.block2 = nn.Sequential(BasicConv(c3, fs, 1), BasicConv(fs, mid, 3))
        self.block3 = nn.Sequential(BasicConv(c4, fs, 1), BasicConv(fs, mid, 3))
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.cls1 = _classifier(mid, fs, num_classes)
        self.cls2 = _classifier(mid, fs, num_classes)
        self.cls3 = _classifier(mid, fs, num_classes)
        self.cls_concat = _classifier(mid * 3, fs, num_classes)

    def forward(self, f2, f3, f4):
        x1 = self.pool(self.block1(f2)).flatten(1)
        x2 = self.pool(self.block2(f3)).flatten(1)
        x3 = self.pool(self.block3(f4)).flatten(1)
        return {
            "logits_pmg1": self.cls1(x1),
            "logits_pmg2": self.cls2(x2),
            "logits_pmg3": self.cls3(x3),
            "logits": self.cls_concat(torch.cat([x1, x2, x3], dim=1)),
        }
