"""机制探针：训练集分得开、测试集塌掉，到底是「均值漂移」还是「方差膨胀」。

背景：同一个模型，训练集类中心余弦最大 0.699，测试集 0.887。
这个差距可以来自两件完全不同的事：

  (a) 均值漂移：同一个类在训练集和测试集上的中心位置不一样（特征跟着域走）
  (b) 方差膨胀：中心没怎么动，但测试样本在中心周围散得更开，邻类就挤进来

两者的对策完全不同：均值漂移要靠类间比较/对齐，方差膨胀要靠一致性/稳健。
所以先量出来，再决定做哪个机制。

定义（对一个类 c）：
    m_c^train, m_c^test         两个划分上的类中心
    s_c^train, s_c^test         类内散度：样本到各自中心的平均距离
    shift_c = ||m_c^train - m_c^test|| / s_c^train     归一化后的中心漂移
    ratio_c = s_c^test / s_c^train                     方差比

若 shift 的中位数明显大于 1，均值漂移是主因；
若 ratio 明显大于 1，方差膨胀是主因。

用法：
    python scripts/analyze_gap.py --checkpoint <ckpt>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import build_dataloaders
from engine.trainer import read_checkpoint, resolve_amp
from model.classifier import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--top-pairs", type=int, default=8)
    return p.parse_args()


@torch.no_grad()
def features(model, cfg, split, device, amp):
    loader = build_dataloaders(cfg, model.backbone, splits=(split,))[split]
    F, Y = [], []
    for b in loader:
        x = b["image"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp):
            f = model.forward_features(x)
        F.append(f.float().cpu())
        Y.append(b["label"])
    return torch.cat(F), torch.cat(Y)


def stats(F, Y, C):
    """返回每类的中心、类内散度、样本数。"""
    d = F.size(1)
    S = torch.zeros(C, d)
    S.index_add_(0, Y, F)
    N = torch.zeros(C).index_add_(0, Y, torch.ones(len(Y)))
    M = S / N.clamp(min=1).view(-1, 1)
    # 类内散度：到本类中心的平均距离
    D = torch.zeros(C)
    D.index_add_(0, Y, (F - M[Y]).norm(dim=1))
    D = D / N.clamp(min=1)
    return M, D, N


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = read_checkpoint(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    amp = resolve_amp(cfg)
    C = cfg["DATASET"]["num_classes"]

    Ftr, Ytr = features(model, cfg, "train", device, amp)
    Fte, Yte = features(model, cfg, "test", device, amp)
    Mtr, Dtr, Ntr = stats(Ftr, Ytr, C)
    Mte, Dte, Nte = stats(Fte, Yte, C)

    shift = (Mtr - Mte).norm(dim=1) / Dtr.clamp(min=1e-6)
    ratio = Dte / Dtr.clamp(min=1e-6)

    print(f"样本数  train={len(Ytr)}  test={len(Yte)}")
    print(f"\n类内散度（到本类中心的平均距离）  train 中位 {Dtr.median():.3f}   test 中位 {Dte.median():.3f}")
    print(f"归一化中心漂移 shift             中位 {shift.median():.2f}   90分位 {shift.quantile(0.9):.2f}")
    print(f"方差比 ratio = s_test/s_train     中位 {ratio.median():.2f}   90分位 {ratio.quantile(0.9):.2f}")

    # 判读
    med_shift, med_ratio = float(shift.median()), float(ratio.median())
    print("\n判读：", end="")
    if med_shift > 1.0 and med_shift > med_ratio - 1:
        print(f"均值漂移为主（shift {med_shift:.2f}），方向应偏向类间比较/对齐")
    elif med_ratio > 1.15:
        print(f"方差膨胀为主（ratio {med_ratio:.2f}），方向应偏向一致性/稳健")
    else:
        print(f"两者都不突出（shift {med_shift:.2f}，ratio {med_ratio:.2f}），"
              f"问题可能在类间距离本身")

    # 最难的那些类对：train 和 test 上的 Fisher 比
    ntr = torch.nn.functional.normalize(Mtr, dim=1)
    cos = ntr @ ntr.t()
    iu = torch.triu_indices(C, C, 1)
    v = cos[iu[0], iu[1]]
    top = v.argsort(descending=True)[: args.top_pairs].tolist()
    a, b = iu[0].tolist(), iu[1].tolist()

    print(f"\n最像的 {args.top_pairs} 对类（按训练集中心余弦排序）：")
    print(f"  {'类对':<12}{'train cos':>10}{'test cos':>10}{'Fisher(train)':>14}{'Fisher(test)':>13}")
    for i in top:
        ca, cb = a[i], b[i]
        def fisher(M, D, x, y):
            return float((M[x] - M[y]).norm() / (D[x] + D[y]).clamp(min=1e-6) * 2)
        def tcos(M, x, y):
            return float(torch.nn.functional.cosine_similarity(M[x], M[y], dim=0))
        print(f"  {ca:3d}-{cb:3d}   {tcos(Mtr, ca, cb):10.3f}{tcos(Mte, ca, cb):10.3f}"
              f"{fisher(Mtr, Dtr, ca, cb):14.3f}{fisher(Mte, Dte, ca, cb):13.3f}")


if __name__ == "__main__":
    main()
