"""Mixup / CutMix。

为什么单独一个文件、还挂在 collate_fn 上：

  混合是**跨样本**的操作（把 batch 里 A 的图混到 B 上），dataset 看不到整个
  batch，而我们又不想去动训练循环。DataLoader 的 collate_fn 正好卡在
  「一批样本已经 transform 完、还没进模型」的位置 —— 输入是一个 list of
  sample dict，输出直接就是 batch。放这里，train_one_epoch 一行都不用改。

产出放在 batch["soft_target"]（[B, num_classes]），由 ClassificationCriterion 识别。

⚠️ 只在**训练** loader 上挂。评估必须是确定性的。
"""

from __future__ import annotations

import torch
from torch.utils.data import default_collate


class MixupCutmix:
    """按 batch 概率性地施加 Mixup 或 CutMix。

    混出来的图同时属于两个类，硬标签不成立，所以产出的是软目标：
        target = lam * onehot(y) + (1 - lam) * onehot(y_perm)
    """

    def __init__(self, mixup: float = 0.0, cutmix: float = 0.0,
                 num_classes: int = 200, p: float = 1.0, switch_p: float = 0.5,
                 manifold: float = 0.0):
        self.mixup = float(mixup)
        self.cutmix = float(cutmix)
        self.num_classes = int(num_classes)
        self.p = float(p)
        self.switch_p = float(switch_p)
        # manifold mixup：不动图像，只把「混哪两个样本、混多少」传给模型，
        # 由 Classifier 在池化后的特征上做插值。它惩罚的是「类间存在非线性空隙」，
        # 和图像层 mixup 是两种不同的正则。
        self.manifold = float(manifold)
        self.enabled = self.mixup > 0 or self.cutmix > 0 or self.manifold > 0

    def _onehot(self, y: torch.Tensor) -> torch.Tensor:
        return torch.zeros(y.size(0), self.num_classes).scatter_(1, y.view(-1, 1), 1.0)

    def _rand_lam(self, alpha: float) -> float:
        return float(torch.distributions.Beta(alpha, alpha).sample())

    def __call__(self, samples: list) -> dict:
        batch = default_collate(samples)
        if not self.enabled or float(torch.rand(1)) > self.p:
            return batch

        x, y = batch["image"], batch["label"]
        perm = torch.randperm(x.size(0))

        # manifold mixup：图像保持干净，只把配对和 lam 传给模型（在特征层混合）
        if self.manifold > 0 and self.mixup <= 0 and self.cutmix <= 0:
            lam = self._rand_lam(self.manifold)
            batch["soft_target"] = (lam * self._onehot(y)
                                    + (1.0 - lam) * self._onehot(y[perm]))
            batch["label_b"] = y[perm]
            batch["mix_perm"] = perm
            batch["mix_lam"] = torch.tensor([lam])
            batch["mix_mode"] = "manifold"
            return batch

        use_mixup = self.mixup > 0 and (self.cutmix <= 0 or float(torch.rand(1)) < self.switch_p)

        # 三流融合时 batch 里还有 head / body 两张局部图。它们必须用**同一个 perm
        # 和同一个 lam** 一起混，否则图像内容和 soft_target 对不上 —— 这种错误不会
        # 报错，只会让准确率悄悄掉几个点。
        keys = [k for k in ("image", "head", "body") if k in batch]

        # 记下这一批用的是哪种混合。CutMix 会把画面抹掉一块，CAM 只能框到残留碎片，
        # 所以 Classifier 会在 CutMix 的 batch 上跳过 CAM 辅助分支（cam_crop / cam_drop）。
        batch["mix_mode"] = "mixup" if use_mixup else "cutmix"

        if use_mixup:
            lam = self._rand_lam(self.mixup)
            for k in keys:
                batch[k] = lam * batch[k] + (1.0 - lam) * batch[k][perm]
        else:
            lam = self._rand_lam(self.cutmix)
            H, W = x.shape[-2:]
            # 让裁剪框面积 ≈ (1-lam)·H·W
            rh = int(H * (1.0 - lam) ** 0.5)
            rw = int(W * (1.0 - lam) ** 0.5)
            cy, cx = int(torch.randint(H, (1,))), int(torch.randint(W, (1,)))
            y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
            x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
            for k in keys:
                v = batch[k].clone()
                v[:, :, y1:y2, x1:x2] = batch[k][perm][:, :, y1:y2, x1:x2]
                batch[k] = v
            # 用**实际贴入的面积**重算 lam，而不是直接用采样值 ——
            # 框贴到边界外时真实面积会更小，不重算的话标签权重是错的。
            lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
        batch["soft_target"] = (lam * self._onehot(y)
                                + (1.0 - lam) * self._onehot(y[perm]))
        # 第二个类别的索引：CAM union 裁剪要用它算 B 的 CAM。
        # 干净 batch（mixup_p < 1 时没被混的那些）不会有这个键。
        batch["label_b"] = y[perm]
        return batch


def build_mixer(tcfg: dict, num_classes: int) -> MixupCutmix | None:
    """按 TRAIN 里的配置建混 mixer；两个 α 都为 0 时返回 None（不挂 collate_fn）。"""
    mixer = MixupCutmix(
        manifold=tcfg.get("manifold_mixup", 0.0),
        mixup=tcfg.get("mixup", 0.0),
        cutmix=tcfg.get("cutmix", 0.0),
        num_classes=num_classes,
        p=tcfg.get("mixup_p", 1.0),
        # 混合批里用 mixup 还是 cutmix 的切换概率。0.5 = 各半；
        # 想让 cutmix 只占混合批的 1/3 就设 0.667。
        switch_p=tcfg.get("switch_p", 0.5),
    )
    return mixer if mixer.enabled else None
