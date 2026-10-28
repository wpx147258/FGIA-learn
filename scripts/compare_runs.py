"""两个配置的配对检验：McNemar + bootstrap 置信区间。

为什么需要它：test 上单个 top1 的标准误是 0.41 个百分点（n=5794, p≈0.89），
比我们想看的大部分差异还大。直接比两个 top1 的小数点第二位，等于在比噪声。

但同一批测试图上的**配对**比较方差小得多：把两张 5794 长的对错向量对齐，
只看「A 对 B 错」和「B 对 A 错」的那些图。这就是 McNemar。

用法：
    python scripts/compare_runs.py a_probs.npy a_labels.npy b_probs.npy b_labels.npy
    python scripts/compare_runs.py --dir dumps  A B      # 自动拼 前缀_probs.npy

先跑 scripts/test_tta.py --dump dumps/<名字> 生成概率文件。
"""

from __future__ import annotations

import argparse
import math

import numpy as np


def load(prefix: str):
    p = np.load(f"{prefix}_probs.npy").astype(np.float32)
    y = np.load(f"{prefix}_labels.npy")
    return p, y


def acc(p, y):
    return float((p.argmax(1) == y).mean() * 100)


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray):
    """返回 (n10, n01, 统计量, p 值)。n10 = A 对 B 错。"""
    n10 = int((correct_a & ~correct_b).sum())
    n01 = int((~correct_a & correct_b).sum())
    n = n10 + n01
    if n == 0:
        return n10, n01, 0.0, 1.0
    z = (abs(n10 - n01) - 1) / math.sqrt(n)
    p = math.erfc(abs(z) / math.sqrt(2))
    return n10, n01, z, p


def bootstrap_delta(correct_a, correct_b, n_boot=5000, seed=0):
    """准确率差值 (B - A) 的 bootstrap 95% 区间，按样本重采样。"""
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    d = correct_b.astype(np.int8) - correct_a.astype(np.int8)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = d[idx].mean(1) * 100
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="概率文件所在目录前缀")
    ap.add_argument("name_a")
    ap.add_argument("name_b", nargs="?", help="省略时只打印 A 的统计")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    def path(name):
        return name if "/" in name or not args.dir else f"{args.dir.rstrip('/')}/{name}"

    pa, ya = load(path(args.name_a))
    ca = pa.argmax(1) == ya
    print(f"A = {args.name_a}: top1 = {acc(pa, ya):.2f}%   (n={len(ya)})")

    if not args.name_b:
        return

    pb, yb = load(path(args.name_b))
    assert (ya == yb).all(), "两个文件的标签顺序不一致，不是同一批测试图"
    cb = pb.argmax(1) == yb
    print(f"B = {args.name_b}: top1 = {acc(pb, yb):.2f}%")
    print(f"差值 B−A = {acc(pb, yb) - acc(pa, ya):+.2f} 个点")

    n10, n01, z, p = mcnemar(ca, cb)
    lo, hi = bootstrap_delta(ca, cb, n_boot=args.n_boot)
    print(f"\n配对（McNemar）:  A对B错 {n10:4d}   A错B对 {n01:4d}   统计量 {z:.2f}   p = {p:.4f}")
    print(f"Δ 的 bootstrap 95% 区间: [{lo:+.2f}, {hi:+.2f}] 个点")
    if p < 0.05 and lo > 0:
        print("结论：B 显著更好")
    elif p < 0.05 and hi < 0:
        print("结论：B 显著更差")
    else:
        print("结论：未检出差异（不能说两者相同，只能说这个样本量下分不开）")


if __name__ == "__main__":
    main()
