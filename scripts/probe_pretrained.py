"""比较不同 ImageNet 预训练权重在 CUB 上的特征质量。

做法：冻结 backbone，只用 GAP 特征训练一个线性分类器，在 val 上报 top-1。
这不是最终精度（没有微调、没有 mixup/裁剪），但能快速排出候选权重的相对高低，
避免每份都花一小时完整微调。

用法：
    python scripts/probe_pretrained.py resnet50.a1_in1k resnet50.a1h_in1k ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.cub import CUBDataset
from dataset.transforms import build_transform, resolve_data_config
from model.backbone import build_backbone
from utils.config import BASE_CONFIG, load_config, resolve_path


def make_loader(root: Path, split: str, transform, bs: int = 64) -> DataLoader:
    ds = CUBDataset(data_root=root, split=split, transform=transform, num_classes=200)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)


@torch.no_grad()
def extract(backbone, loader, device):
    feats, labels = [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            f = backbone(x)
        feats.append(f.float().cpu())
        labels.append(batch["label"])
    return torch.cat(feats), torch.cat(labels)


def linear_probe(tr_x, tr_y, va_x, va_y, device, epochs: int = 60) -> float:
    tr_x, va_x = tr_x.to(device), va_x.to(device)
    tr_y, va_y = tr_y.to(device), va_y.to(device)
    # 标准化：不同权重的特征尺度不同，不统一的话学习率没法共用
    mu, sd = tr_x.mean(0, keepdim=True), tr_x.std(0, keepdim=True).clamp(min=1e-6)
    tr_x, va_x = (tr_x - mu) / sd, (va_x - mu) / sd

    clf = torch.nn.Linear(tr_x.shape[1], 200).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = tr_x.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            F.cross_entropy(clf(tr_x[idx]), tr_y[idx]).backward()
            opt.step()
        sched.step()
    with torch.no_grad():
        pred = clf(va_x).argmax(1)
    return (pred == va_y).float().mean().item() * 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config([BASE_CONFIG], ["DATASET.image_size=448"])
    root = resolve_path(cfg["DATASET"]["root"])

    print(f"{'backbone':<36}{'val top-1':>12}{'特征维度':>12}")
    print("-" * 62)
    for name in args.names:
        bb = build_backbone(name, pretrained=True).to(device).eval()
        # 探针必须用**评估** transform：随机增强会把噪声混进特征，压低压倒性差异
        tf = build_transform(resolve_data_config(bb), 448, train=False)
        tr_x, tr_y = extract(bb, make_loader(root, "train", tf), device)
        va_x, va_y = extract(bb, make_loader(root, "val", tf), device)
        acc = linear_probe(tr_x, tr_y, va_x, va_y, device)
        print(f"{name:<36}{acc:11.2f}%{tr_x.shape[1]:12d}", flush=True)
        del bb
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
