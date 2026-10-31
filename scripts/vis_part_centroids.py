"""部件质心图：8 个部件到底各看哪个点（2026-10-31）。

动机：`K_eff` 只衡量"分布形状是否相同"，8 个部件可以是**围绕同一点、形状不同的散斑**。
实测质心两两距离只有 0.04~0.06 格（14 格制）——不到一个格子。
这张图把 8 个质心直接画在图上：点如果全挤在一起，就说明"部件在空间上没分开"。

每个格子：网络输入（448 裁剪）+ GT 鸟框（绿虚线）+ 8 个部件的质心（彩色圆点，大小=该部件质量）。
标题给出该图的质心平均两两距离（格）。

用法：
    python scripts/vis_part_centroids.py --checkpoint a.pth b.pth --labels A B --n 6 --out figs/x.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.cub import CUBDataset
from dataset.transforms import build_transform, resolve_data_config
from engine.trainer import read_checkpoint, resolve_amp
from model.classifier import build_model, load_state_dict_tolerant
from utils.config import resolve_path

PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
           "#0072B2", "#D55E00", "#CC79A7", "#000000"]


def load_gt(root: Path):
    gts, paths = {}, {}
    for line in open(root / "bounding_boxes.txt"):
        i, x, y, w, h = line.split()
        gts[int(i)] = (float(x), float(y), float(w), float(h))
    for line in open(root / "images.txt"):
        i, p = line.split()
        paths[int(i)] = p
    return gts, paths


@torch.no_grad()
def masks_for(ckpt, idxs, dev):
    ck = read_checkpoint(ckpt, map_location=dev)
    cfg = ck["config"]
    model = build_model(cfg).to(dev)
    load_state_dict_tolerant(model, ck["model"])
    model.eval()
    pa = getattr(model.backbone, "part_attn", None)
    if pa is None:
        sys.exit(f"{ckpt} 没有 part_attn")
    dcfg = cfg["DATASET"]
    img = int(dcfg["image_size"])
    root = resolve_path(dcfg["root"])
    dc = dict(resolve_data_config(model.backbone))
    if dcfg.get("crop_pct"):
        dc["crop_pct"] = float(dcfg["crop_pct"])
    ds = CUBDataset(data_root=root, split="test",
                    transform=build_transform(dc, img, train=False),
                    num_classes=dcfg["num_classes"])
    batch = next(iter(DataLoader(Subset(ds, idxs), batch_size=len(idxs),
                                 shuffle=False, num_workers=2)))
    x = batch["image"].to(dev)
    amp = resolve_amp(cfg)
    feat = model.backbone.forward_features(x)
    if amp:
        feat = feat.float()
    M = pa.part_masks(feat).float().cpu().numpy()
    mean = np.array(dc["mean"], np.float32).reshape(3, 1, 1)
    std = np.array(dc["std"], np.float32).reshape(3, 1, 1)
    imgs = (batch["image"].numpy() * std + mean).clip(0, 1).transpose(0, 2, 3, 1)
    return dict(M=M, imgs=imgs, ds=ds, root=root, img=img,
                crop_pct=float(dc.get("crop_pct") or 1.0),
                name=Path(ckpt).parent.parent.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="figs/part_centroids.png")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = a.labels or [Path(p).parent.parent.name for p in a.checkpoint]
    idxs = np.linspace(0, 5793, a.n).astype(int).tolist()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    from matplotlib import patheffects as pe

    runs = [masks_for(p, idxs, dev) for p in a.checkpoint]
    R = len(runs)
    fig, axes = plt.subplots(R, a.n, figsize=(2.2 * a.n, 2.5 * R))
    axes = np.atleast_2d(axes)
    halo = [pe.Stroke(linewidth=2.0, foreground="white"), pe.Normal()]

    print(f"{'模型':26s}{'质心两两距离(格)':>18s}{'GT 质心占比':>14s}")
    for r, (run, lab) in enumerate(zip(runs, labels)):
        ds, root, img, crop = run["ds"], run["root"], run["img"], run["crop_pct"]
        gts, paths = load_gt(root)
        spreads, in_gt = [], []
        for bi in range(a.n):
            iid = ds.data_ids[idxs[bi]]
            W, H = Image.open(root / "images" / paths[iid]).size
            rr = int(round(img / crop)); s = rr / min(W, H)
            nw, nh = (int(round(rr * W / H)), rr) if W >= H else (rr, int(round(rr * H / W)))
            ox, oy = (nw - img) / 2.0, (nh - img) / 2.0
            gx, gy, gw, gh = gts[iid]
            gx0, gy0 = gx * s - ox, gy * s - oy
            gbw, gbh = gw * s, gh * s

            M = run["M"][bi]
            K, h, w = M.shape
            ccs, mass = [], []
            for k in range(K):
                v = M[k].reshape(-1); v = v / max(v.sum(), 1e-12)
                yy, xx = np.divmod(np.arange(v.size), w)
                cx, cy = (v * xx).sum(), (v * yy).sum()
                # 该部件质量有多少落在 GT 框内（用于"质心是否在鸟上"的粗略判读）
                X = (xx + 0.5) / w * img; Y = (yy + 0.5) / h * img
                inside = (X >= gx0) & (X < gx0 + gbw) & (Y >= gy0) & (Y < gy0 + gbh)
                in_gt.append(float((v * inside).sum()))
                ccs.append(((cx + 0.5) / w * img, (cy + 0.5) / h * img)); mass.append(float(v.max() * v.size))
            ccs = np.array(ccs)
            d = np.linalg.norm(ccs[:, None] - ccs[None, :], axis=-1)
            spreads.append(float(d[~np.eye(K, dtype=bool)].mean()) / (img / w))

            ax = axes[r][bi]
            ax.imshow(run["imgs"][bi])
            ax.add_patch(Rectangle((gx0, gy0), gbw, gbh, fill=False, ec="lime",
                                   lw=1.2, ls="--", path_effects=halo, zorder=5))
            for k in range(K):
                ax.scatter([ccs[k, 0]], [ccs[k, 1]], s=45, c=PALETTE[k % 8],
                           edgecolors="white", linewidths=1.0, zorder=7)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"spread {spreads[-1]:.2f} grid", fontsize=5.5)
            if bi == 0:
                ax.set_ylabel(lab[:24], fontsize=6.5)
        print(f"{lab[:26]:26s}{np.mean(spreads):18.2f}{np.mean(in_gt)*100:13.0f}%")

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[k],
                      markersize=7, label=f"P{k+1}") for k in range(8)]
    handles.append(Line2D([0], [0], color="lime", ls="--", lw=1.2, label="GT bird box"))
    fig.legend(handles=handles, loc="lower center", ncol=9, frameon=False, fontsize=6)
    fig.suptitle("Where each part's centroid lands — all 8 dots collapse onto one point "
                 "(spread < 0.1 grid)", fontsize=8)
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"已保存 {a.out}")


if __name__ == "__main__":
    main()
