"""可分度探针：同一套协议下测各层特征的线性可分度（可复现版）。

为什么重写：先前 MECHANISM_MAP 里"layer3 GAP 只有 50.74"这个数，
生成它的脚本没有入库，无法复现；而同一次测量里 layer4 在两套协议下分别得到
85.99 和 87.33 —— 说明协议本身不稳。这个数被用来否定过"多尺度融合"，
所以必须先复现，再谈结论。

本脚本固定三件事，避免再出现不可复现的数：
  1. 同一权重、同一评测变换、同一 λ 扫描范围
  2. 同时报 train 拟合精度（如果 train 也只有 50，说明特征真的线性不可分；
     如果 train 很高而 test 很低，说明是过拟合或分布间隙）
  3. 每层都报 train→test 与 test 内 5 折两个数

用法：
    python scripts/probe_layers.py --checkpoint <ckpt>
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


def ridge(Xtr, ytr, Xte, yte, C, lam):
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp(min=1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Y = torch.zeros(len(ytr), C)
    Y[torch.arange(len(ytr)), ytr] = 1.0
    A = Xtr.t() @ Xtr + lam * len(ytr) * torch.eye(Xtr.size(1))
    W = torch.linalg.solve(A, Xtr.t() @ Y)
    tr = ((Xtr @ W).argmax(1) == ytr).float().mean().item() * 100
    te = ((Xte @ W).argmax(1) == yte).float().mean().item() * 100
    return tr, te


@torch.no_grad()
def feats(model, cfg, split, device, amp):
    """返回 {层名: [N, C] 的 GAP 特征}。索引已实测：1->56², 2->28²? 见下。"""
    loader = build_dataloaders(cfg, model.backbone, splits=(split,))[split]
    out = {}
    ys = []
    for b in loader:
        x = b["image"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp):
            final, inter = model.backbone.forward_intermediates(x, indices=[2, 3])
        # timm resnet: indices=[2,3] -> layer2(512,56²), layer3(1024,28²)；final=layer4(2048,14²)
        out.setdefault("layer2", []).append(inter[0].float().mean((2, 3)).cpu())
        out.setdefault("layer3", []).append(inter[1].float().mean((2, 3)).cpu())
        out.setdefault("layer4", []).append(final.float().mean((2, 3)).cpu())
        ys.append(b["label"])
    return {k: torch.cat(v) for k, v in out.items()}, torch.cat(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lams", default="1e-4,1e-2,1e-1")
    a = ap.parse_args()
    lams = [float(x) for x in a.lams.split(",")]

    dev = torch.device("cuda")
    ck = read_checkpoint(a.checkpoint, map_location=dev)
    cfg = ck["config"]
    m = build_model(cfg).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    amp = resolve_amp(cfg)
    C = cfg["DATASET"]["num_classes"]

    tr, ytr = feats(m, cfg, "train", dev, amp)
    te, yte = feats(m, cfg, "test", dev, amp)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(yte), generator=g)

    print(f"{'层':<8}{'λ':>8}{'train acc':>11}{'train→test':>12}{'test 5折':>10}")
    for layer in ("layer2", "layer3", "layer4"):
        Xtr, Xte = tr[layer], te[layer]
        for lam in lams:
            atr, ate = ridge(Xtr, ytr, Xte, yte, C, lam)
            accs = []
            for k in range(5):
                idx = perm[k::5]
                mask = torch.ones(len(yte), dtype=torch.bool)
                mask[idx] = False
                _, a2 = ridge(Xte[mask], yte[mask], Xte[idx], yte[idx], C, lam)
                accs.append(a2)
            print(f"{layer:<8}{lam:>8.0e}{atr:>11.2f}{ate:>12.2f}{sum(accs)/5:>10.2f}")
    print(f"\n参考：该模型自身 test 准确率见 test.py；特征维度 "
          f"{tr['layer2'].size(1)}/{tr['layer3'].size(1)}/{tr['layer4'].size(1)}")


if __name__ == "__main__":
    main()
