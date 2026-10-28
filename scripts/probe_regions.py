"""部位信息探针：带同面积随机框对照。

背景：早先做过一次「用 GT 头部框对 layer4 特征做池化」的探针，得到 81.55%，
全局平均池化是 88.61%，于是我们判断「判别信息在全局、不在部位」，并据此
放弃了整条部位路线。这个判断有两个硬伤，本脚本就是来补对照的：

  1. 没有同面积的随机框对照。头框只占原图 5.4%（14×14 图上约 20/196 格）。
     如果随便框一块同样大的区域也掉到 81%，那 81.55 说明的是「小块区域信息少」，
     而不是「头部信息少」。
  2. layer4 单格感受野已经接近整图，框内池化出来的向量并不是「部位描述子」，
     它照样混着全身和背景。所以这个探针本来就不能回答「部位有没有信息」。

顺便补一个「中心化余弦」：类中心都共享一个很大的「鸟」的均值方向，
直接算余弦会被整体抬高。扣掉全局均值再算，才是真正的类间相似度。

用法：
    python scripts/probe_regions.py --checkpoint <ckpt>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.cub import CUBDataset
from dataset.transforms import build_transform, resolve_data_config
from engine.trainer import read_checkpoint, resolve_amp
from model.classifier import build_model
from torch.utils.data import DataLoader
from utils.config import resolve_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ridge", type=float, default=1e-2, help="线性探针的正则强度")
    return p.parse_args()


@torch.no_grad()
def extract(model, cfg, split, device, amp, rng_seed=0):
    """一次前向，同时算出五种池化结果（不保留整张特征图，否则显存/内存会爆）。"""
    dcfg = cfg["DATASET"]
    root = resolve_path(dcfg["root"])
    data_cfg = resolve_data_config(model.backbone)
    ds = CUBDataset(data_root=root, split=split,
                    transform=build_transform(data_cfg, dcfg["image_size"], train=False),
                    num_classes=dcfg["num_classes"], use_parts=True)
    loader = DataLoader(ds, batch_size=cfg["EVAL"]["batch_size"], shuffle=False,
                        num_workers=dcfg["num_workers"], pin_memory=True)

    torch.manual_seed(rng_seed)
    acc = {k: [] for k in ("gap", "head", "rand", "out_head", "body")}
    labels = []
    g = 14
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp):
            fm = model.backbone.forward_features(x)
        fm = fm.float().cpu()
        B = fm.size(0)
        hm, bm = [], []
        for i in range(B):
            img_id = batch["image_id"][i].item()
            p = ds.imgs_paths[img_id]
            W, H = Image.open(p).size
            hb, bb = ds._part_boxes(img_id, W, H)
            hm.append(box_mask(hb, W, H))
            bm.append(box_mask(bb, W, H))
        hm, bm = torch.stack(hm), torch.stack(bm)
        rm = random_like(hm, g)
        full = torch.ones(B, g * g)
        acc["gap"].append(masked_pool(fm, full))
        acc["head"].append(masked_pool(fm, hm))
        acc["rand"].append(masked_pool(fm, rm))
        acc["out_head"].append(masked_pool(fm, 1 - hm))
        acc["body"].append(masked_pool(fm, bm))
        labels.append(batch["label"])
    return {k: torch.cat(v) for k, v in acc.items()}, torch.cat(labels)


def random_like(mask, g):
    """和头部框同样高宽的随机框（同面积对照）。"""
    out = torch.zeros_like(mask)
    for i in range(len(mask)):
        rr, cc = torch.nonzero(mask[i].view(g, g), as_tuple=True)
        h = int(rr.max() - rr.min() + 1)
        w = int(cc.max() - cc.min() + 1)
        r = torch.randint(0, max(1, g - h + 1), (1,)).item()
        c = torch.randint(0, max(1, g - w + 1), (1,)).item()
        out[i].view(g, g)[r:r + h, c:c + w] = 1.0
    return out


def box_mask(box, W, H, g: int = 14):
    """框 -> [g*g] 的 0/1 掩码。"""
    m = torch.zeros(g * g)
    c0 = max(0, min(int(box[0] / W * g), g - 1))
    c1 = max(c0 + 1, min(int(np.ceil(box[2] / W * g)), g))
    r0 = max(0, min(int(box[1] / H * g), g - 1))
    r1 = max(r0 + 1, min(int(np.ceil(box[3] / H * g)), g))
    m.view(g, g)[r0:r1, c0:c1] = 1.0
    return m


def masked_pool(fm, mask):
    """fm [N,C,g,g]，mask [N,g*g] -> [N,C]。空掩码时退回全局平均。"""
    N, C, g, _ = fm.shape
    m = mask.view(N, 1, g, g)
    cnt = m.sum(dim=(2, 3)).clamp(min=1.0)
    return (fm * m).sum(dim=(2, 3)) / cnt


def ridge_probe(Xtr, ytr, Xte, yte, num_classes, lam):
    """闭式岭回归探针：一次求解，比 SGD 稳，也不受学习率影响。"""
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp(min=1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Y = torch.zeros(len(ytr), num_classes)
    Y[torch.arange(len(ytr)), ytr] = 1.0
    A = Xtr.t() @ Xtr + lam * len(ytr) * torch.eye(Xtr.size(1))
    W = torch.linalg.solve(A, Xtr.t() @ Y)
    pred = (Xte @ W).argmax(1)
    return (pred == yte).float().mean().item() * 100


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = read_checkpoint(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    amp = resolve_amp(cfg)

    tr, ytr = extract(model, cfg, "train", device, amp, rng_seed=args.seed)
    te, yte = extract(model, cfg, "test", device, amp, rng_seed=args.seed + 1)
    C = cfg["DATASET"]["num_classes"]

    names = {
        "gap": "全局 GAP",
        "head": "头部框（5.4% 面积）",
        "rand": "同面积随机框",
        "out_head": "除头部以外",
        "body": "身体框（29% 面积）",
    }

    print(f"探针：闭式岭回归（λ={args.ridge}），训练 {len(ytr)} / 测试 {len(yte)}\n")
    print(f"  {'池化区域':<22}{'test top1':>10}")
    for k, name in names.items():
        a = ridge_probe(tr[k], ytr, te[k], yte, C, args.ridge)
        print(f"  {name:<22}{a:>10.2f}")

    print("\n类中心余弦（原始 vs 扣掉全局均值方向）：")
    for tag, X, y in (("train", tr["gap"], ytr), ("test", te["gap"], yte)):
        cent = torch.zeros(C, X.size(1))
        cent.index_add_(0, y, X)
        cnt = torch.zeros(C).index_add_(0, y, torch.ones(len(y)))
        cent = cent / cnt.clamp(min=1).view(-1, 1)
        raw = torch.nn.functional.normalize(cent, dim=1)
        ctr = torch.nn.functional.normalize(cent - cent.mean(0, keepdim=True), dim=1)
        iu = torch.triu_indices(C, C, 1)
        vr = (raw @ raw.t())[iu[0], iu[1]]
        vc = (ctr @ ctr.t())[iu[0], iu[1]]
        print(f"  {tag}: 原始 最大 {vr.max():.3f} (>0.9 有 {(vr > .9).sum().item()} 对)  "
              f"| 中心化 最大 {vc.max():.3f} 均值 {vc.mean():.3f}")


if __name__ == "__main__":
    main()
