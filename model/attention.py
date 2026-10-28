"""注意力模块：K 张可学习的注意力图 + 部位分类。

和 CAM 的本质区别（这是当初决定要写它的原因）：

    CAM       cam = Σ_j W_cj · F_jhw        —— 通道组合由**分类头权重 W_c 定死**，只有 1 张
    注意力模块 A_k = sigmoid(Σ_j U_kj · F_jhw) —— 有 K 组**各自独立学习**的通道组合

所以它不是"把 CAM 变成 K 张"，而是"让模型自己学 K 种看这张图的方式"。
K 组不同的通道组合可以理解成 K 个不同的部位探测器（有的敏感于喙的形状，有的敏感于
羽毛纹理……），然后各自去特征图上做一次加权池化，各自分类。

监督信号只有类别标签：**不告诉模型"第 k 个部位该看哪里"，只要求"每个部位都得认出整只鸟"**。

⚠️ 只在训练时用。评估走 Classifier.forward 的原路径，推理开销为零。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionAux(nn.Module):
    """K 张注意力图 + 部位分类头。

    参数：
        in_channels  backbone 特征图的通道数（ResNet50 是 2048）
        num_classes  类别数
        num_parts    K，注意力图的数量
        weight       辅助损失在总损失里的权重
    """

    def __init__(self, in_channels: int, num_classes: int,
                 num_parts: int = 32, weight: float = 0.5):
        super().__init__()
        self.weight = float(weight)
        self.num_parts = int(num_parts)
        # 1×1 卷积 = 对每个部位学一组通道权重（U_k），输出第 k 张注意力图的"原始分数"
        self.attn = nn.Conv2d(in_channels, num_parts, kernel_size=1)
        # 部位分类头：K 个部位共用一组权重 —— 因为每个部位都应该能独立认出整只鸟
        self.part_head = nn.Linear(in_channels, num_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat [B, C, h, w] -> 部位 logits [B, K, num_classes]"""
        A = self.attn(feat).sigmoid()                            # [B,K,h,w] 每个数在 (0,1)
        # 空间维 L1 归一化：让池化是「凸组合」，尺度稳定，且和 GAP 同量级
        A = A / A.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)

        # 注意力池化：第 k 个部位的向量 = 特征图按 A_k 加权平均
        f = torch.einsum("bkhw,bchw->bkc", A, feat)              # [B,K,C]
        return self.part_head(f)                                 # [B,K,num_classes]

    @torch.no_grad()
    def diversity(self, feat: torch.Tensor) -> dict:
        """诊断：K 张注意力图有没有分化。

        如果 32 张图全都塌在同一片区域，这个模块就没在做它该做的事 ——
        那是个该早点发现的问题，所以留一个可以直接调的诊断。
        返回平均的两两 IoU（越低越分散）。
        """
        A = self.attn(feat).sigmoid()
        A = A / A.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        B, K, h, w = A.shape
        flat = A.reshape(B, K, h * w)
        inter = torch.bmm(flat, flat.transpose(1, 2))            # [B,K,K]
        union = flat.sum(-1).unsqueeze(2) + flat.sum(-1).unsqueeze(1) - inter
        iou = (inter / union.clamp(min=1e-6))
        eye = torch.eye(K, device=A.device, dtype=torch.bool).unsqueeze(0)
        return {"iou_offdiag": float(iou.masked_fill(eye, 0).sum() / (B * K * (K - 1)))}


class AttentionAuxAdapter:
    """把 AttentionAux 包成和 CamAux 一样的调用约定，方便 Classifier 统一调度。

    CamAux 的签名是 (model, x, feat, logits, y) -> dict；这里对齐它。
    """

    def __init__(self, module: AttentionAux):
        self.module = module

    def __call__(self, model, x, feat, logits, y):
        return {"logits_parts": self.module(feat)}


def build_attention_aux(tcfg: dict, mcfg: dict, in_channels: int, num_classes: int):
    w = float(tcfg.get("attn_weight", 0.0) or 0.0)
    if w <= 0:
        return None
    return AttentionAux(
        in_channels=in_channels,
        num_classes=num_classes,
        num_parts=int(mcfg.get("num_parts", 32)),
        weight=w,
    )
