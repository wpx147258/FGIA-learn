"""评估指标。

纯函数 + 两个累计器，不做注册表 —— 指标就是普通函数，直接 import 调用即可，
包一层「按名字取出函数再调用」没有任何收益。
"""

from __future__ import annotations

import torch


class AverageMeter:
    """按样本数加权的累计平均。

    直接对一个 batch 一个 batch 的标量求算术平均是错的：最后一个 batch
    往往不满，它的权重和其它 batch 不一样。所以必须按样本数加权。
    """

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> dict[str, float]:
    """top-k 准确率，**单位是百分比**（不是 0~1 的小数）。

    logits: [B, num_classes]
    target: [B]
    """
    max_k = min(max(topk), logits.size(1))
    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)   # [B, max_k]
    pred = pred.t()                                                  # [max_k, B]
    correct = pred.eq(target.view(1, -1).expand_as(pred))            # [max_k, B]

    out = {}
    for k in topk:
        k = min(k, logits.size(1))
        out[f"top{k}"] = correct[:k].reshape(-1).float().sum().item() / target.size(0) * 100.0
    return out


class PerClassAccuracy:
    """跨 batch 累计每类的正确数 / 总数。

    为什么不逐 batch 算准确率再平均：
      1) 每个类的样本数不一样，「每类准确率的平均」（macro）和「按样本加权的
         平均」（micro/top1）是两个不同的量，混着算会得到一个谁也不认识的数；
      2) batch 内类分布本身在抖，逐 batch 平均会把这个抖动引进指标。

    CUB 的 val 只有 599 张、每类 2~3 张，所以 macro 和 top1 会很接近。
    但跑 test（5794 张）时两者会分开，那时 macro 更能反映「有没有被忽略的类」。
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.correct = torch.zeros(num_classes, dtype=torch.long)
        self.total = torch.zeros(num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(dim=1).detach().cpu()
        target = target.detach().cpu()
        self.total += torch.bincount(target, minlength=self.num_classes)
        self.correct += torch.bincount(target[pred == target], minlength=self.num_classes)

    @torch.no_grad()
    def compute(self) -> dict:
        """返回：
            per_class_acc   Tensor[num_classes]，百分比，无样本的类是 nan
            per_class_count Tensor[num_classes]
            macro           float，每类准确率的平均（忽略空类）
        """
        acc = torch.full((self.num_classes,), float("nan"))
        nonempty = self.total > 0
        acc[nonempty] = self.correct[nonempty].float() / self.total[nonempty] * 100.0
        return {
            "per_class_acc": acc,
            "per_class_count": self.total.clone(),
            "macro": acc[nonempty].mean().item() if bool(nonempty.any()) else float("nan"),
        }
