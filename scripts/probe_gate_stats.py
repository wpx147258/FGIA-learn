"""门控读出的机制诊断：门到底有没有打开？

对 `MODEL.tl_readout=gate` 的权重，统计每个样本、每个 token 的门控值
`g_s = sigmoid(MLP(z_s))`：

  * 初始状态 g≈4.5e-5（读出的输出≈GAP）；
  * 如果训练后 g 仍然≈0 → 模型选择「不要部件」，等于退回 GAP（这不是失败，
    而是说明门控给出了拒绝的答案）；
  * 如果部分 token 的门明显打开（>0.1）、部分保持关闭 → 说明它在**按样本选择**
    相信哪几个部件，这正是门控想做的事。

同时报告 `|pooled − GAP|` 的相对幅度，直接看门控把输出推离 GAP 多远。

用法：
    python scripts/probe_gate_stats.py --checkpoint <ckpt> [--n 96]
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
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = read_checkpoint(a.checkpoint, map_location=dev)
    cfg = ck["config"]
    amp = resolve_amp(cfg)
    m = build_model(cfg).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    tl = m.tokenlearner
    if tl is None or getattr(tl, "gate_mlp", None) is None:
        raise SystemExit("这个 checkpoint 不是 gate 读出（tokenlearner.gate_mlp 不存在）")

    dcfg = cfg["DATASET"]
    dc = dict(resolve_data_config(m.backbone))
    if dcfg.get("crop_pct"):
        dc["crop_pct"] = float(dcfg["crop_pct"])
    ds = CUBDataset(resolve_path(dcfg["root"]), "test",
                    build_transform(dc, dcfg["image_size"], train=False), dcfg["num_classes"])
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=dcfg["num_workers"])

    rows, rel = [], []
    seen = 0
    with torch.no_grad():
        for b in dl:
            x = b["image"].to(dev)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                f = m.backbone.forward_features(x)
                fm = m._to_feature_map(f)
                with torch.autocast("cuda", enabled=False):
                    ff = fm.float()
                    t = tl.proj(ff).flatten(2).transpose(1, 2)
                    w = tl.score(t).softmax(dim=1)
                    z = torch.einsum("bns,bnc->bsc", w, ff.flatten(2).transpose(1, 2))
                    g = torch.sigmoid(tl.gate_mlp(z)).squeeze(-1)      # [B,S]
                    gap = ff.mean(dim=(2, 3))
                    out = gap + (g[..., None] * z).sum(dim=1)
                    rel.append(((out - gap).norm(dim=1) / gap.norm(dim=1).clamp_min(1e-6))
                               .mean().item())
            rows.append(g.cpu())
            seen += x.shape[0]
            if seen >= a.n:
                break
    G = torch.cat(rows)[:a.n].numpy()
    print(f"checkpoint: {a.checkpoint}")
    print(f"  样本 {len(G)}, S={G.shape[1]}")
    print(f"  门控值   mean={G.mean():.4f}  median={np.median(G):.4f}  max={G.max():.4f}")
    print(f"  每样本最大门  mean={G.max(1).mean():.4f}  median={np.median(G.max(1)):.4f}")
    print(f"  关闭的门（g<0.01）占比 {(G < 0.01).mean():.3f}")
    print(f"  打开的门（g>0.10）占比 {(G > 0.10).mean():.3f}")
    print(f"  |pooled − GAP| / |GAP|  {np.mean(rel):.4f}   (0 = 完全退回 GAP)")


if __name__ == "__main__":
    main()
