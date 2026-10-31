"""多模型部件框对照图：行 = 配置，列 = 同一批图（2026-10-31）。

和 `vis_part_boxes.py` 的区别：那个一次只画一个 checkpoint；这个把**几个部件注意力配置**
画在同一批图上逐行对照，格式与 `figs/cam_boxes_best.png` 一致（行=模型、列=图片），
便于直接看"改一个机制，框的几何怎么变"。

每个格子画：
  · 8 个部件的「累计 80% 质量」外接框（8 色）
  · GT 鸟框（绿色虚线）
  · 标题：`框面积均值% | 框内 GT 覆盖率均值%`（8 个部件的平均）

用法：
    python scripts/vis_part_boxes_multi.py \
        --checkpoint a/swa_f0.5.pth b/swa_f0.5.pth c/last.pth \
        --labels 标准PA 紧致 四项完整 --n 8 --out figs/part_boxes_multi.png
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
    with open(root / "bounding_boxes.txt") as f:
        for line in f:
            iid, x, y, w, h = line.split()
            gts[int(iid)] = (float(x), float(y), float(w), float(h))
    with open(root / "images.txt") as f:
        for line in f:
            iid, p = line.split()
            paths[int(iid)] = p
    return gts, paths


@torch.no_grad()
def run_masks(ckpt_path, idxs, dev):
    ck = read_checkpoint(ckpt_path, map_location=dev)
    cfg = ck["config"]
    model = build_model(cfg).to(dev)
    load_state_dict_tolerant(model, ck["model"])
    model.eval()
    pa = getattr(model.backbone, "part_attn", None)
    if pa is None:
        sys.exit(f"{ckpt_path} 没有 part_attn")
    dcfg = cfg["DATASET"]
    img_size = int(dcfg["image_size"])
    root = resolve_path(dcfg["root"])
    data_cfg = dict(resolve_data_config(model.backbone))
    if dcfg.get("crop_pct"):
        data_cfg["crop_pct"] = float(dcfg["crop_pct"])
    ds = CUBDataset(data_root=root, split="test",
                    transform=build_transform(data_cfg, img_size, train=False),
                    num_classes=dcfg["num_classes"])
    batch = next(iter(DataLoader(Subset(ds, idxs), batch_size=len(idxs),
                                 shuffle=False, num_workers=2)))
    amp = resolve_amp(cfg)
    x = batch["image"].to(dev)
    feat = model.backbone.forward_features(x).float()
    masks = pa.part_masks(feat).float().cpu().numpy()
    mean = np.array(data_cfg["mean"], np.float32).reshape(3, 1, 1)
    std = np.array(data_cfg["std"], np.float32).reshape(3, 1, 1)
    imgs = (batch["image"].numpy() * std + mean).clip(0, 1).transpose(0, 2, 3, 1)
    return dict(masks=masks, imgs=imgs, ds=ds, root=root, img_size=img_size,
                crop_pct=float(data_cfg.get("crop_pct") or 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="figs/part_boxes_multi.png")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = a.labels or [Path(p).parent.parent.name for p in a.checkpoint]
    idxs = np.linspace(0, 5793, a.n).astype(int).tolist()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib import patheffects as pe

    runs = [run_masks(p, idxs, dev) for p in a.checkpoint]
    R = len(runs)
    fig, axes = plt.subplots(R, a.n, figsize=(2.15 * a.n, 2.4 * R))
    axes = np.atleast_2d(axes)
    halo = [pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()]

    print("每个格子：8 个部件的「80% 质量外接框」均值  |  框内 GT 覆盖率均值")
    for r, (run, lab) in enumerate(zip(runs, labels)):
        ds, root, img_size, crop_pct = run["ds"], run["root"], run["img_size"], run["crop_pct"]
        gts, paths = load_gt(root)
        areas, covs = [], []
        for bi in range(a.n):
            iid = ds.data_ids[idxs[bi]]
            W, H = Image.open(root / "images" / paths[iid]).size
            rr = int(round(img_size / crop_pct))
            s = rr / min(W, H)
            nw, nh = (int(round(rr * W / H)), rr) if W >= H else (rr, int(round(rr * H / W)))
            ox, oy = (nw - img_size) / 2.0, (nh - img_size) / 2.0
            gx, gy, gw, gh = gts[iid]
            gx0, gy0 = gx * s - ox, gy * s - oy
            gbw, gbh = gw * s, gh * s

            M = run["masks"][bi]
            K, h, w = M.shape
            ax = axes[r][bi]
            ax.imshow(run["imgs"][bi])
            a_area, a_cov = [], []
            for k in range(K):
                v = M[k].reshape(-1)
                o = np.argsort(-v)
                c = np.cumsum(v[o])
                n = int(np.searchsorted(c, 0.8 * c[-1]) + 1)
                ys, xs = np.divmod(o[:n], w)
                x0, y0 = xs.min() / w * img_size, ys.min() / h * img_size
                x1, y1 = (xs.max() + 1) / w * img_size, (ys.max() + 1) / h * img_size
                area = (x1 - x0) * (y1 - y0) / (img_size * img_size)
                inter = max(0.0, min(x1, gx0 + gbw) - max(x0, gx0)) * max(0.0, min(y1, gy0 + gbh) - max(y0, gy0))
                a_area.append(area); a_cov.append(inter / max((x1 - x0) * (y1 - y0), 1e-9))
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       ec=PALETTE[k % len(PALETTE)], lw=1.1,
                                       path_effects=halo, zorder=6))
            ax.add_patch(Rectangle((gx0, gy0), gbw, gbh, fill=False, ec="lime",
                                   lw=1.3, ls="--", path_effects=halo, zorder=5))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{np.mean(a_area)*100:.0f}% | GT {np.mean(a_cov)*100:.0f}%", fontsize=5.5)
            if bi == 0:
                ax.set_ylabel(lab[:22], fontsize=6.5)
            areas += a_area; covs += a_cov
        print(f"[{lab}] 框面积均值 {np.mean(areas)*100:5.1f}%   框内 GT 覆盖率 {np.mean(covs)*100:5.1f}%")

    fig.suptitle("Part-attention boxes: 8 parts (colored) vs GT bird box (green) — "
                 "same 8 images, three mechanisms", fontsize=8)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"已保存 {a.out}")


if __name__ == "__main__":
    main()
