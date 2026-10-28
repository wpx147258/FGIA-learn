"""诊断：8 个部件 token 里有没有「平均不到」的互补信息？

问题（PART_ATTENTION_FINDINGS §14）：mean 读出 `pooled = mean(z_1..z_S)` 与 GAP 同构，
所以把权重图调得再准、分化得再好，也可能只是重新加权。要判断「换 concat 读出」值不值得，
得先回答一个纯推理问题：

    **同一批 token，分类头能分别利用它们时，准确率会不会显著高于只用它们的均值？**

做法（不训练、只拟合线性分类器）：
  1. 用给定 checkpoint 在 train/test 子集上抽三类特征：
        gap    —— backbone 的全局平均特征 [C]
        mean   —— S 个 token 的平均 [C]（= replace+mean 读出喂给分类头的东西）
        concat —— S 个 token 直接拼接 [S*C]
        tok_k  —— 单个 token [C]（逐个拟合，看单部件有多强）
  2. 在 train 子集上拟合 sklearn LogisticRegression，在 test 子集上报 top1。

判读：
  * concat ≫ mean → 部件里有平均掉的信息，concat 读出值得做；
  * concat ≈ mean  ≈ gap → 部件没有互补信息，换读出也救不了这条线（要换机制）。

⚠️ 这是**事后**探针：token 是在 mean 目标下训出来的，concat 模型训练后 token 会不同，
所以 concat 的数字是下界，不是预言。它的用处是判断「值不值得花一轮 GPU 去训 concat」。

用法：
    python scripts/probe_token_probe.py --checkpoint <ckpt> --n-train 3000 --n-test 3000
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


def collect(m, dev, amp, cfg, split, n, bs):
    dcfg = cfg["DATASET"]
    dc = dict(resolve_data_config(m.backbone))
    if dcfg.get("crop_pct"):
        dc["crop_pct"] = float(dcfg["crop_pct"])
    ds = CUBDataset(resolve_path(dcfg["root"]), split,
                    build_transform(dc, dcfg["image_size"], train=False), dcfg["num_classes"])
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=dcfg["num_workers"])
    feats = {"gap": [], "mean": [], "tok": []}
    ys, seen = [], 0
    tl = m.tokenlearner
    with torch.no_grad():
        for b in dl:
            x = b["image"].to(dev)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                f = m.backbone.forward_features(x)
                fm = m._to_feature_map(f)
                gap = m.backbone.forward_head(f)
                if tl is not None:
                    with torch.autocast("cuda", enabled=False):
                        ff = fm.float()
                        t = tl.proj(ff).flatten(2).transpose(1, 2)
                        w = tl.score(t).softmax(dim=1)                  # [B,N,S]
                        z = torch.einsum("bns,bnc->bsc", w, ff.flatten(2).transpose(1, 2))
                else:
                    z = gap[:, None, :]
            feats["gap"].append(gap.float().cpu())
            feats["mean"].append(z.float().mean(1).cpu())
            feats["tok"].append(z.float().cpu())
            ys.append(b["label"])
            seen += x.shape[0]
            if seen >= n:
                break
    out = {k: torch.cat(v)[:n].numpy() for k, v in feats.items()}
    return out, torch.cat(ys)[:n].numpy()


def fit_eval(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
    clf.fit(sc.transform(Xtr), ytr)
    return float((clf.predict(sc.transform(Xte)) == yte).mean() * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = read_checkpoint(a.checkpoint, map_location=dev)
    cfg = ck["config"]; amp = resolve_amp(cfg)
    m = build_model(cfg).to(dev); m.load_state_dict(ck["model"]); m.eval()
    print(f"checkpoint: {a.checkpoint}")
    tr, ytr = collect(m, dev, amp, cfg, "train", a.n_train, a.batch_size)
    te, yte = collect(m, dev, amp, cfg, "test", a.n_test, a.batch_size)
    s = tr["tok"].shape[1]
    res = {}
    res["gap"] = fit_eval(tr["gap"], ytr, te["gap"], yte)
    res["mean"] = fit_eval(tr["mean"], ytr, te["mean"], yte)
    res["concat"] = fit_eval(tr["tok"].reshape(len(tr["tok"]), -1), ytr,
                             te["tok"].reshape(len(te["tok"]), -1), yte)
    per = []
    for k in range(s):
        per.append(fit_eval(tr["tok"][:, k], ytr, te["tok"][:, k], yte))
    print(f"  线性探针 top1（train {len(ytr)} / test {len(yte)}）")
    for k, v in res.items():
        print(f"    {k:<7} {v:.2f}")
    print(f"    单 token  逐部件 {['%.1f' % v for v in per]}  (均值 {np.mean(per):.2f})")
    print(f"    concat − mean = {res['concat'] - res['mean']:+.2f} 个点"
          f"   mean − gap = {res['mean'] - res['gap']:+.2f}")


if __name__ == "__main__":
    main()
