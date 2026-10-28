"""在测试集上算 200×200 的混淆矩阵，看错误到底聚不聚在少数几个格子。

动机：我们一直用"类中心余弦 > 0.9"描述难类，但余弦只说明两个类在特征空间里近，
不说明模型真的把它们互相认错，也不说明错的方向。混淆矩阵能直接回答三件事：

  1. 错误是集中（少数 off-diagonal 格子）还是弥散（摊在 200×200 里）
  2. 难对是不是对称的（A→B 和 B→A 都错，还是单向错）
  3. Top-1 错误里有百分之几落在最好的 K 个格子里

用法：
    python scripts/confusion_matrix.py --checkpoint outputs/<run>/<ts>/swa_f0.5.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import build_dataloaders
from engine.trainer import read_checkpoint, resolve_amp
from model.classifier import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--top", type=int, default=15, help="打印前 N 个错误格子")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = read_checkpoint(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    C = cfg["DATASET"]["num_classes"]

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = build_dataloaders(cfg, model.backbone, splits=(args.split,))[args.split]
    amp = resolve_amp(cfg)

    cm = torch.zeros(C, C, dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"]
            if amp is not None:
                with torch.autocast(device_type="cuda", dtype=amp):
                    logits = model(x)
            else:
                logits = model(x)
            pred = logits.float().argmax(1).cpu()
            for t, q in zip(y.tolist(), pred.tolist()):
                cm[t, q] += 1

    n = cm.sum().item()
    correct = cm.diag().sum().item()
    err = cm.clone()
    err.fill_diagonal_(0)
    total_err = err.sum().item()
    print(f"{args.split}: n={n}  top1={correct / n * 100:.2f}%  错误 {total_err} 个")

    # 错误集中度：前 K 个格子占全部错误的比例
    flat = err.flatten().sort(descending=True).values
    for k in (5, 10, 20, 50, 100):
        print(f"  前 {k:3d} 个格子占全部错误的 {flat[:k].sum().item() / total_err * 100:5.1f}%"
              f"   （随机期望 {k / (C * (C - 1)) * 100:.3f}%）")

    print(f"\n前 {args.top} 个错误格子（真类 → 预测类）：")
    vals, idx = flat[: args.top], err.flatten().topk(args.top).indices
    for v, i in zip(vals.tolist(), idx.tolist()):
        a, b = divmod(i, C)
        ra = cm[a].sum().item()
        rb = cm[b].sum().item()
        back = cm[b, a].item()
        print(f"  {a:3d} → {b:3d}   {v:3d} 次   该类共 {ra:3d} 张(错 {v / ra * 100:4.1f}%)"
              f"   反向 {b}→{a}: {back} 次")


if __name__ == "__main__":
    main()
