"""用训练集统计的通道相关性，把 layer4 的 2048 个通道聚成 K 组，输出通道重排 perm。

动机：torch 的通道索引是任意的，"连续切分" 没有任何语义依据；相关性聚类让同一组的
通道响应相似，部件描述子才可能对应同一类证据。输出 perm.npy 给 MODEL.pa_perm 用。

用法：python scripts/make_channel_groups.py --checkpoint <ckpt> --k 4 --n 2000 \
        --out outputs/grouping/perm_k4.npy
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
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    dev = torch.device("cuda")
    ck = read_checkpoint(args.checkpoint, map_location=dev)
    cfg = ck["config"]; amp = resolve_amp(cfg)
    model = build_model(cfg).to(dev); model.load_state_dict(ck["model"]); model.eval()
    dcfg = cfg["DATASET"]
    ds = CUBDataset(resolve_path(dcfg["root"]), "train",
                    build_transform(resolve_data_config(model.backbone), dcfg["image_size"], train=False),
                    dcfg["num_classes"])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=dcfg["num_workers"])
    feats = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(dev)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                f = model.backbone.forward_features(x)
            feats.append(f.float().mean(dim=(2, 3)).cpu())
            if sum(t.shape[0] for t in feats) >= args.n:
                break
    X = torch.cat(feats)[: args.n].numpy()
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
    C = np.corrcoef(X, rowvar=False)                      # [2048,2048]
    C = np.clip(np.nan_to_num(C, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    # 相关矩阵含负值，不能直接当谱聚类的相似度图；转成距离后用层次聚类
    D = np.clip((1.0 - (C + C.T) / 2.0) / 2.0, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    from sklearn.cluster import AgglomerativeClustering
    lab = AgglomerativeClustering(n_clusters=args.k, metric="precomputed",
                                  linkage="average").fit_predict(D)
    perm = np.concatenate([np.where(lab == c)[0] for c in range(args.k)])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, perm)
    print("cluster sizes:", [int((lab == c).sum()) for c in range(args.k)])
    print("saved", args.out, "n_samples", X.shape[0])
    # 组内 vs 组间平均相关性，作为「分组是否比连续切分更有依据」的量化
    def avg_corr(idx_a, idx_b):
        block = C[np.ix_(idx_a, idx_b)]
        if idx_a is idx_b or np.array_equal(idx_a, idx_b):
            m = ~np.eye(len(idx_a), dtype=bool)
            return float(block[m].mean())
        return float(block.mean())
    within = np.mean([avg_corr(np.where(lab == c)[0], np.where(lab == c)[0]) for c in range(args.k)])
    cont = np.array_split(np.arange(2048), args.k)
    within_c = np.mean([avg_corr(g, g) for g in cont])
    print(f"组内平均相关: 聚类 {within:.3f}  vs  连续切分 {within_c:.3f}")


if __name__ == "__main__":
    main()
