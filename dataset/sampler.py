"""P×K 批采样器：每批抽 P 个类，每类 K 张图。

为什么成对判别非要它不可：API-Net 的配对是**批内**按特征距离找最近邻。
我们原来的 loader 每批 16 张随机图、散在 200 个类里，同类对的期望数量约等于 0，
于是"同类最近邻"根本不存在的，配对会退化成随机。

副作用必须写清楚，否则后面看结果会归错因：
每批只覆盖 P 个类，类别组成比随机采样窄得多，BN 统计和 mixup 的配对分布
都会跟着变。所以「开了成对判别」这一组和主配方的对比里，其实有两个变量：
采样方式 + 成对损失。要更干净就得再补一个「只开 P×K、不开成对」的对照。
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import Sampler


class PKBatchSampler(Sampler[list[int]]):
    """按 dataset 的标签序列生成 P×K 的批次。

    参数：
        labels  与 dataset 下标一一对应的标签（0-based）
        p       每批的类数
        k       每类抽几张
        seed    随机种子；每个 epoch 用 seed+epoch 重新洗牌，续训也能复现
    """

    def __init__(self, labels, p: int, k: int, seed: int = 0):
        if p < 2 or k < 2:
            raise ValueError(f"P×K 至少要 2×2（收到 {p}×{k}）："
                             "同类对和异类对都要求每批有多个样本")
        self.p, self.k = int(p), int(k)
        self.seed = int(seed)
        self.epoch = 0

        by_class: dict[int, list[int]] = defaultdict(list)
        for idx, y in enumerate(labels):
            by_class[int(y)].append(idx)
        # 少于 2 张的类配不出同类对，直接排除（CUB 每类至少 20 张，实际不会触发）
        self.by_class = {c: v for c, v in by_class.items() if len(v) >= 2}
        if len(self.by_class) < self.p:
            raise ValueError(f"只有 {len(self.by_class)} 个可用类，不够每批 {self.p} 个")
        self.classes = sorted(self.by_class)
        self.n_batches = max(1, len(labels) // (self.p * self.k))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _class_stream(self, g: torch.Generator):
        """无限产出打乱后的类 id；一轮用完再重新洗牌。"""
        while True:
            perm = torch.randperm(len(self.classes), generator=g).tolist()
            for c in perm:
                yield self.classes[c]

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        stream = self._class_stream(g)
        for _ in range(self.n_batches):
            batch: list[int] = []
            for _ in range(self.p):
                c = next(stream)
                pool = self.by_class[c]
                # 够 k 张就无放回（否则同一张图会以两个不同的增广进同一批，
                # 同类最近邻可能选中它自己的另一个版本，配对退化成噪声对）；
                # 不够 k 张时才回落到有放回。
                if len(pool) >= self.k:
                    pick = torch.randperm(len(pool), generator=g)[:self.k].tolist()
                else:
                    pick = torch.randint(len(pool), (self.k,), generator=g).tolist()
                batch.extend(pool[i] for i in pick)
            yield batch

    def __len__(self) -> int:
        return self.n_batches
