"""可视化 TokenLearner / PartAttention 的部件注意力图（定性检查「分化成什么样」）。

对每张图输出 1 + S 个面板：原图（叠 CAM 或全局响应）与 S 张部件权重图。
每张图标题给出该图的平均熵与两两 JS，方便和 `probe_token_stats.py` 的数值对上。

用法：
  python scripts/vis_token_maps.py --checkpoint <ckpt> --out figs/tl_maps.png [--n 2]
"""

from __future__ import annotations
import argparse, math, sys
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


def js_pairwise(a):
    """a: [N,S] 概率分布（每列和为 1）-> 两两 JS 均值（numpy 版）。"""
    a = np.clip(a, 1e-12, None)
    m = (a[:, :, None] + a[:, None, :]) / 2.0
    kl1 = (a[:, :, None] * (np.log(a[:, :, None]) - np.log(np.clip(m, 1e-12, None)))).sum(0)
    js = 0.5 * (kl1 + kl1.T)
    s = js.shape[0]
    off = 1.0 - np.eye(s)
    return float((js * off).sum() / (s * (s - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = read_checkpoint(a.checkpoint, map_location=dev)
    cfg = ck["config"]; amp = resolve_amp(cfg)
    m = build_model(cfg).to(dev); m.load_state_dict(ck["model"]); m.eval()
    tl = m.tokenlearner
    pa = getattr(m.backbone, "part_attn", None)
    dcfg = cfg["DATASET"]
    dc = dict(resolve_data_config(m.backbone))
    if dcfg.get("crop_pct"): dc["crop_pct"] = float(dcfg["crop_pct"])
    ds = CUBDataset(resolve_path(dcfg["root"]), a.split,
                    build_transform(dc, dcfg["image_size"], train=False), dcfg["num_classes"])
    dl = DataLoader(ds, batch_size=a.n, shuffle=False, num_workers=dcfg["num_workers"])
    batch = next(iter(dl))
    x = batch["image"].to(dev); y = batch["label"].to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=amp, enabled=amp is not None):
        f = m.backbone.forward_features(x)
        fm = m._to_feature_map(f).float()
        logits = m.head(m.backbone.forward_head(f))
        if tl is not None:
            t = tl.proj(fm).flatten(2).transpose(1, 2)
            w = tl.score(t).softmax(dim=1)                   # [B,N,S] 对位置归一化
        else:
            if pa is None:
                raise SystemExit("checkpoint 里既没有 tokenlearner 也没有 part_attn")
            masks = pa.part_masks(fm)                        # [B,K,h,w]
            w = masks.flatten(2)                             # [B,K,N]（每张掩码未归一化）
            w = (w / w.sum(-1, keepdim=True).clamp_min(1e-6)).transpose(1, 2)
    h = int(w.shape[1] ** 0.5); S = w.shape[-1]
    W = w.float().transpose(1, 2).reshape(-1, S, h, h)       # [B,S,h,h]
    W = torch.nn.functional.interpolate(W.reshape(-1, 1, h, h), size=fm.shape[-1] * 1,
                                        mode="bilinear", align_corners=False)
    W = W.reshape(x.shape[0], S, fm.shape[-1], fm.shape[-1]).cpu().numpy()

    mean = np.array(dc["mean"], dtype=np.float32); std = np.array(dc["std"], dtype=np.float32)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(x.shape[0], S + 1, figsize=(2.0 * (S + 1), 2.2 * x.shape[0]))
    axes = np.atleast_2d(axes)
    for i in range(x.shape[0]):
        im = np.clip(x[i].cpu().permute(1, 2, 0).numpy() * std + mean, 0, 1)
        pred = int(logits[i].argmax()); truth = int(y[i])
        axes[i, 0].imshow(im); axes[i, 0].set_title(
            f"image (pred {'OK' if pred == truth else 'X'})", fontsize=8); axes[i, 0].axis("off")
        ww = W[i].reshape(S, -1)
        ww = ww / ww.sum(1, keepdims=True)
        ent = -(ww * np.log(np.clip(ww, 1e-9, None))).sum(1).mean()
        js = js_pairwise(ww.T)                               # 传入 [N,S]
        for s in range(S):
            ax = axes[i, s + 1]
            ax.imshow(im)
            c = W[i, s]
            c = (c - c.min()) / (c.max() - c.min() + 1e-9)
            ax.imshow(np.ma.masked_where(c < 0.55, c), cmap="jet", alpha=0.55)
            ax.set_title(f"part {s+1}", fontsize=8); ax.axis("off")
        axes[i, S].set_xlabel(f"ent={ent:.2f} (ln k, k≈{math.exp(ent):.1f})  JS={js:.2f}",
                              fontsize=8)
    fig.suptitle(f"part maps — {Path(a.checkpoint).parts[-3]}", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("saved", a.out)


if __name__ == "__main__":
    main()
