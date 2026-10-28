"""强视图生成（张量级，训练专用）。

为什么是张量级而不是 dataset 级：
  审查指出两处致命问题。
  1) 如果为了给强视图开 RandAugment 去设 TRAIN.rand_augment，弱视图会**同时**被改，
     基座立刻从 89.40 变成别的配方 —— 双变量。
  2) dataset 里每次调用 Compose 都独立采样裁剪/翻转，两次调用拿到的裁剪不同，
     一致性损失会悄悄变成「裁剪不变性」，和设计意图不是一个东西。
  所以强视图必须从**已经过 mixup/cutmix 的 batch['image']** 上做张量级几何变换，
  这样它自动继承同一组 mix 标签，也不会碰弱视图的训练变换。

变换：随机旋转 ±deg、剪切 ±shear、平移 ±trans，边界填充；可选随机擦除。
都在归一化后的张量上做，擦除填 0 = 填 ImageNet 均值色。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def make_strong_view(x: torch.Tensor, deg: float = 15.0, shear: float = 10.0,
                     trans: float = 0.10, erase_p: float = 0.0,
                     erase_area: float = 0.15) -> torch.Tensor:
    """x [B,3,H,W]（已归一化）-> 同形状的强扰动版本。逐样本独立采样参数。"""
    B = x.size(0)
    dev = x.device
    rad = math.pi / 180.0

    ang = (torch.rand(B, device=dev) * 2 - 1) * deg * rad
    sh = (torch.rand(B, device=dev) * 2 - 1) * shear * rad
    tx = (torch.rand(B, device=dev) * 2 - 1) * trans
    ty = (torch.rand(B, device=dev) * 2 - 1) * trans

    cos, sin = torch.cos(ang), torch.sin(ang)
    tan = torch.tan(sh)
    # 先剪切再旋转：A = R @ Sh
    R = torch.stack([torch.stack([cos, -sin], -1),
                     torch.stack([sin, cos], -1)], -2)          # [B,2,2]
    Sh = torch.stack([torch.stack([torch.ones_like(tan), tan], -1),
                      torch.stack([torch.zeros_like(tan), torch.ones_like(tan)], -1)], -2)
    A = R @ Sh

    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, :, :2] = A
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    out = F.grid_sample(x, grid, align_corners=False, padding_mode="border")

    if erase_p > 0:
        B, _, H, W = out.shape
        hit = torch.rand(B, device=dev) < erase_p
        for i in torch.nonzero(hit).flatten().tolist():
            h = int(H * math.sqrt(erase_area) * (0.5 + torch.rand(1).item()))
            w = int(W * math.sqrt(erase_area) * (0.5 + torch.rand(1).item()))
            h, w = max(1, min(h, H)), max(1, min(w, W))
            y0 = int(torch.randint(0, H - h + 1, (1,)).item())
            x0 = int(torch.randint(0, W - w + 1, (1,)).item())
            out[i, :, y0:y0 + h, x0:x0 + w] = 0.0
    return out
