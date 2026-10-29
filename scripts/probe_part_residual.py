"""部件注意力的残差分支到底起没起作用？

背景（2026-10-29 复查发现的结构事实）：
    `PartAttention.hook` 把部件描述子 `y`（形如 [B,C]）以 `y[:,:,None,None]`
    加回 layer4 特征图 —— 等于**在每个空间位置加同一个通道向量**。
    由于 GAP 是线性的，送进分类器的就是 `pooled = GAP + y`：
    一个逐通道的加性偏移，**与空间权重无关**。
    部件掩码只影响 `y` 的取值，不影响池化时各位置的权重。

本探针量化三件事：
  1. 残差幅度：`‖y‖ / ‖GAP‖`，以及 `y` 与 `GAP` 的余弦（相近 = 冗余，负 = 反证）；
  2. 关掉残差（bypass）后的准确率变化——这是它「是否影响决策」的直接度量；
  3. 逐样本的预测翻转率（有多少张图的 argmax 被这个偏移改变了）。

用法：
    python scripts/probe_part_residual.py --checkpoint <ckpt> [--n 2000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.cub import CUBDataset
from dataset.transforms import build_transform, resolve_data_config
from engine.trainer import read_checkpoint, resolve_amp
from model.classifier import build_model
from utils.config import resolve_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = read_checkpoint(a.checkpoint, map_location=dev)
    cfg = ck["config"]
    amp = resolve_amp(cfg)
    m = build_model(cfg).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    pa = getattr(m.backbone, "part_attn", None)
    if pa is None:
        raise SystemExit("这个 checkpoint 没有 part_attn")

    dcfg = cfg["DATASET"]
    dc = dict(resolve_data_config(m.backbone))
    if dcfg.get("crop_pct"):
        dc["crop_pct"] = float(dcfg["crop_pct"])
    ds = CUBDataset(resolve_path(dcfg["root"]), "test",
                    build_transform(dc, dcfg["image_size"], train=False), dcfg["num_classes"])
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=dcfg["num_workers"])

    correct_on = correct_off = 0
    n = 0
    rel_norm, cos_gap, dl_abs = [], [], []   # dl_abs 这里改为「非均匀成分占比」
    with torch.no_grad():
        for b in dl:
            x = b["image"].to(dev)
            y = b["label"].to(dev)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                logits_on = m(x)
            pa.bypass = True
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                logits_off = m(x)
            pa.bypass = False
            res = getattr(pa, "_last_res", None)
            if res is not None:
                # 残差在 logit 空间的分量：head(y) 与 head(GAP) 的关系
                with torch.no_grad():
                    dl = (logits_on.float() - logits_off.float())          # [B,C] 残差带来的 logit 改变
                    lg = logits_off.float()
                    rel_norm.append((dl.norm(dim=1) / lg.norm(dim=1).clamp_min(1e-6)).mean().item())
                    cos_gap.append(torch.nn.functional.cosine_similarity(dl, lg, dim=1).mean().item())
                    # 「均匀偏置」成分：把 dl 减去逐样本均值后还剩多少
                    dl_c = dl - dl.mean(dim=1, keepdim=True)
                    dl_abs.append((dl_c.norm(dim=1) / dl.norm(dim=1).clamp_min(1e-6)).mean().item())
            correct_on += (logits_on.argmax(1) == y).sum().item()
            correct_off += (logits_off.argmax(1) == y).sum().item()
            if res is not None:
                # GAP：用与 PartAttention 同一层特征（layer4 输出，未加残差时）
                pass
            n += x.shape[0]
            if n >= a.n:
                break
    print(f"checkpoint: {a.checkpoint}")
    print(f"  样本 {n}")
    print(f"  带残差   top1 = {correct_on / n * 100:.2f}%")
    print(f"  关掉残差 top1 = {correct_off / n * 100:.2f}%")
    print(f"  差值            = {(correct_on - correct_off) / n * 100:+.2f} 个点")
    print(f"  ‖Δlogit‖/‖logit‖          = {np.mean(rel_norm):.4f}")
    print(f"  cos(Δlogit, logit)         = {np.mean(cos_gap):.4f}  (接近 1 = 只是整体缩放)")
    print(f"  去掉类间均值后残余占比     = {np.mean(dl_abs):.4f}  (接近 0 = 对所有类等量偏置)")


if __name__ == "__main__":
    main()
