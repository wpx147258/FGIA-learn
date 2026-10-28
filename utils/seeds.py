"""随机种子。

不做这一步的话，两次消融实验的数据增强顺序都不一样，指标差异无法归因。
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """固定 random / numpy / torch 的种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # benchmark=True 会让 cudnn 自动挑最快的卷积算法，但每次跑的算法可能不同，
        # 结果就不可复现了。要可复现就得关掉它。
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id: int) -> None:
    """给每个 DataLoader worker 设种子。

    必须传这个函数（配合 DataLoader 的 generator 参数），否则多 worker 下
    每个 worker 的增强随机性是继承父进程的同一份状态，等价于所有 worker
    用同一个随机序列 —— 增强的多样性会打折扣。
    """
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
