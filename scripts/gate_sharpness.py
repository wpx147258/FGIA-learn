"""诊断门 D2：权重空间的"锐度"到底跟 test 准确率有没有关系。

为什么先做这个再做 SAM：
  SAM 的假设是「解太尖锐 → 测试差」。但这个项目里从没测过锐度，
  而且基线已经用了 SWA（末段权重平均，本身就是一种平坦化）。
  如果在我们已有的十几个 checkpoint 上，锐度和 test top1 根本不相关，
  那么 SAM 就算实现完美也打不到点上 —— 这笔 4.5 小时的账可以先不花。

做法（不训练，只推理）：
  对每个 checkpoint 的 swa_f0.5.pth：
    w  = 权重
    w' = w + ρ · w / (||w|| + eps)     # SAM 原式的逐张量相对扰动
  分别算 test 单视图 top1，记 Δ = top1(w) − top1(w')。
  Δ 越大 = 这个解越"尖锐"。
  然后看 Δ 与 top1(w) 在多个 run 之间是否相关（Spearman）。

判读（预注册）：
  |Spearman| < 0.3 或符号不一致 → 锐度不是准确率的载体，J1(SAM) 直接 DROP。
  显著负相关（越尖锐越差）才值得继续。

用法：
    python scripts/gate_sharpness.py [--rho 0.05] [--limit 0]
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
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=0, help="只测前 N 个（0=全部）")
    p.add_argument("--seeds", type=int, default=3,
                   help="随机方向取几个并取平均（单个方向方差太大，跨 checkpoint 不可比）")
    p.add_argument("--mode", default="random", choices=["random", "scale"],
                   help="random = 固定相对范数的随机方向（真正测锐度）; scale = 全体乘 (1+rho)，BN 会吸收，只作对照")
    return p.parse_args()


@torch.no_grad()
def top1(model, loader, device, amp):
    ok = n = 0
    for b in loader:
        x = b["image"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp):
            lo = model(x)
        ok += (lo.float().argmax(1).cpu() == b["label"]).sum().item()
        n += len(b["label"])
    return ok / n * 100


@torch.no_grad()
def perturb_(model, rho, mode="random"):
    """SAM 原式 e = ρ·w/‖w‖：逐张量的相对扰动，等价于 w *= (1+ρ)。

    只扰动维度 >= 2 的权重矩阵，不动 BN 的 weight/bias 和一维参数。
    返回备份，用于精确恢复。
    """
    backup = {}
    for name, p in model.named_parameters():
        backup[name] = p.detach().clone()
        if p.ndim >= 2:
            if mode == "scale":
                p.mul_(1.0 + rho)      # BN 会吸收，作对照
            else:
                u = torch.randn_like(p)
                p.add_(rho * p.norm() * u / (u.norm() + 1e-12))   # 每层等相对幅度
    return backup


def restore_(model, backup):
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(backup[name])


def main():
    args = parse_args()
    device = torch.device("cuda")
    runs = sorted(Path("outputs").glob("*/2*/swa_f0.5.pth"))
    if args.limit:
        runs = runs[: args.limit]
    print(f"共 {len(runs)} 个 checkpoint，ρ = {args.rho}\n")

    rows = []
    for path in runs:
        name = path.parent.parent.name
        ck = read_checkpoint(path, map_location=device)
        cfg = ck["config"]
        model = build_model(cfg).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        amp = resolve_amp(cfg)
        loader = build_dataloaders(cfg, model.backbone, splits=("test",))["test"]

        a = top1(model, loader, device, amp)
        drops = []
        for k in range(args.seeds):
            torch.manual_seed(1234 + k)        # 所有 checkpoint 共用同一组方向
            backup = perturb_(model, args.rho, args.mode)
            b = top1(model, loader, device, amp)
            restore_(model, backup)
            drops.append(a - b)
        d = float(np.mean(drops))
        rows.append((name, a, d))
        print(f"  {name:28s} top1={a:6.2f}  Δ(mean of {args.seeds})={d:+6.2f}  "
              f"各方向 {[round(x,2) for x in drops]}")

    acc = np.array([r[1] for r in rows])
    dlt = np.array([r[3] for r in rows])
    # Spearman（无 scipy，手算秩相关）
    def rank(x):
        order = x.argsort()
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(x))
        return r
    rx, ry = rank(acc), rank(dlt)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    print(f"\nSpearman(Δ锐度, test top1) = {rho:+.3f}   n={len(rows)}")
    print(f"判据 |ρ|<0.3 → {'DROP SAM（锐度不是载体）' if abs(rho) < 0.3 else '保留 SAM 候选'}")


if __name__ == "__main__":
    main()
