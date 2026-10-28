"""核对 TokenLearner / 部件模块的注意力统计量，并把「冗余」与「分散」分开度量。

对每张图、每张权重图 w_s（在 N 个位置上和为 1）计算：
  * entropy       每张图的空间熵（自然对数；均匀 = ln N）
  * overlap_raw   两两内积（TokenLearner 的 div 项用的就是这个）
  * cos_center    先减空间均值再算余弦（去掉「公共的整体分布」）
  * js            两两 JS 散度（上界 ln2）
  * iou_top5      各取响应最高 5% 的格子，算支撑集 IoU

用法：python scripts/probe_token_stats.py --checkpoint <ckpt> [--n 200]
"""

from __future__ import annotations
import argparse, math, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.cub import CUBDataset
from dataset.transforms import build_transform, resolve_data_config
from engine.trainer import read_checkpoint, resolve_amp
from model.classifier import build_model
from utils.config import resolve_path


def pair_metrics(w):
    """w: [B,N,S] -> 各指标的均值（只统计 s<m 的对）。"""
    b, n, s = w.shape
    eps = 1e-9
    ent = -(w * w.clamp_min(eps).log()).sum(1)                 # [B,S]
    ov = w.transpose(1, 2) @ w                               # [B,S,S] 两两在位置上的内积
    wc = w - w.mean(1, keepdim=True)
    wcn = wc / wc.norm(dim=1, keepdim=True).clamp_min(eps)
    cosc = wcn.transpose(1, 2) @ wcn                         # [B,S,S]
    # JS: 0.5 KL(p||m) + 0.5 KL(q||m)
    m = (w[:, :, :, None] + w[:, :, None, :]) / 2.0           # [B,N,S,S]
    kl1 = (w[:, :, :, None] * (w[:, :, :, None].clamp_min(eps).log()
                               - m.clamp_min(eps).log())).sum(1)
    kl2 = (w[:, :, None, :] * (w[:, :, None, :].clamp_min(eps).log()
                               - m.clamp_min(eps).log())).sum(1)
    js = 0.5 * (kl1 + kl2)                                     # [B,S,S]
    k = max(1, int(0.05 * n))
    idx = w.topk(k, dim=1).indices                             # [B,k,S]
    mask = torch.zeros_like(w).scatter(1, idx, 1.0)            # [B,N,S]
    inter = mask.transpose(1, 2) @ mask                       # [B,S,S]
    union = mask.sum(1)[:, :, None] + mask.sum(1)[:, None, :] - inter
    iou = inter / union.clamp_min(1)
    off = torch.ones(s, s, device=w.device) - torch.eye(s, device=w.device)
    sel = off.bool()
    return (ent.mean().item(), ov[:, sel].mean().item(), cosc[:, sel].mean().item(),
            js[:, sel].mean().item(), iou[:, sel].mean().item(), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = read_checkpoint(a.checkpoint, map_location=dev)
    cfg = ck["config"]; amp = resolve_amp(cfg)
    m = build_model(cfg).to(dev); m.load_state_dict(ck["model"]); m.eval()
    tl = m.tokenlearner
    if tl is None:
        raise SystemExit("这个 checkpoint 没有 tokenlearner")
    dcfg = cfg["DATASET"]
    dc = dict(resolve_data_config(m.backbone))
    if dcfg.get("crop_pct"): dc["crop_pct"] = float(dcfg["crop_pct"])
    ds = CUBDataset(resolve_path(dcfg["root"]), "test",
                    build_transform(dc, dcfg["image_size"], train=False), dcfg["num_classes"])
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=dcfg["num_workers"])
    rows = []; seen = 0
    with torch.no_grad():
        for b in dl:
            x = b["image"].to(dev)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                f = m.backbone.forward_features(x)
                fm = m._to_feature_map(f)
                t = tl.proj(fm).flatten(2).transpose(1, 2)
                w = tl.score(t).softmax(dim=1)                 # [B,N,S]
            rows.append(pair_metrics(w.float()))
            seen += x.shape[0]
            if seen >= a.n: break
    mean = [sum(r[i] for r in rows) / len(rows) for i in range(5)]
    n_pos = rows[0][5]
    print(f"checkpoint: {a.checkpoint}")
    print(f"  N={n_pos} 位置, S={w.shape[-1]} 张图, 均匀熵上界 lnN={math.log(n_pos):.3f}")
    print(f"  entropy      {mean[0]:.4f}   (归一化 {mean[0]/math.log(n_pos):.3f})")
    print(f"  overlap_raw  {mean[1]:.4f}   (1/N={1/n_pos:.4f} 表示毫无重叠)")
    print(f"  cos_center   {mean[2]:.4f}   (0 表示去掉公共成分后完全不像)")
    print(f"  JS           {mean[3]:.4f}   (上界 ln2={math.log(2):.3f})")
    print(f"  iou_top5%    {mean[4]:.4f}")


if __name__ == "__main__":
    main()
