"""定量 + 定性分析 CAM 裁剪分支：框落在哪、和 GT 框重合多少、框质量与裁剪分支正确率的关系。

背景：CAM 裁剪分支是本项目唯一被证明有效的训练期机制（去掉它 −0.85~−1.19），
但它的框来自主分支特征图上的 CAM。一旦换池化/注意力模块改变了特征几何，框就会变——
这条耦合一直没有被量化过。本脚本回答三件事：
  1. 框有多大、落在鸟身上吗（与 GT bbox 的 IoU / 精确率 / 召回率）；
  2. 框越准，裁剪分支越准吗（按 IoU 分层看 crop 分支的 top-1）；
  3. 定性看一组图：原图 + CAM 热力图 + CAM 框（绿）+ GT 框（红）。

用法：
  python scripts/probe_cam_box.py --checkpoint <ckpt> [--n 200] [--fig figs/cam_boxes.png]
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.cub import CUBDataset
from dataset.transforms import build_transform, resolve_data_config
from engine.trainer import read_checkpoint, resolve_amp
from model.cam_aug import cam_map, crop_and_resize, mask_to_box, peak_mask
from model.classifier import build_model
from utils.config import resolve_path


def load_gt_boxes(root: Path, W: int, H: int):
    """CUB 的 bounding_boxes.txt：image_id x y w h（像素，原图坐标）-> 归一化框。"""
    gt = {}
    with open(root / "bounding_boxes.txt") as f:
        for line in f:
            p = line.split()
            if len(p) != 5: continue
            i, x, y, w, h = int(p[0]), *map(float, p[1:])
            gt[i] = (x / W, y / H, (x + w) / W, (y + h) / H)
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--fig", default="")
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = read_checkpoint(a.checkpoint, map_location=dev)
    cfg = ck["config"]; amp = resolve_amp(cfg)
    m = build_model(cfg).to(dev); m.load_state_dict(ck["model"]); m.eval()
    dcfg = cfg["DATASET"]; root = resolve_path(dcfg["root"])
    dc = dict(resolve_data_config(m.backbone))
    if dcfg.get("crop_pct"): dc["crop_pct"] = float(dcfg["crop_pct"])
    tf = build_transform(dc, dcfg["image_size"], train=False)
    ds = CUBDataset(root, a.split, tf, dcfg["num_classes"])
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=dcfg["num_workers"])

    # 原图尺寸（用于把像素框归一化）；按需读取并缓存，避免一次性打开 5794 张图
    with open(root / "images.txt") as f:
        paths = {int(l.split()[0]): l.split()[1] for l in f if l.strip()}
    from PIL import Image
    _size_cache = {}
    def img_size(i):
        if i not in _size_cache:
            _size_cache[i] = Image.open(root / "images" / paths[i]).size
        return _size_cache[i]
    gt_px = {}
    with open(root / "bounding_boxes.txt") as f:
        for line in f:
            q = line.split()
            if len(q) != 5: continue
            i, x, y, w, h = int(q[0]), *map(float, q[1:])
            gt_px[i] = (x, y, w, h)

    rows, figs = [], []
    seen = 0
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(dev); y = batch["label"].to(dev)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                feat = m.backbone.forward_features(x)
                logits = m.head(m.backbone.forward_head(feat))
            fm = m._to_feature_map(feat).float()
            w_ = m.head.cam_weight(fm.shape[1]).detach()
            # 用真实标签算 CAM（训练期 CAM 分支就是这么做的）
            cam = cam_map(fm, w_, y)
            boxes = mask_to_box(peak_mask(cam, float(cfg["TRAIN"].get("cam_alpha_crop", 0.2))).detach())
            patch = crop_and_resize(x, boxes, int(cfg["DATASET"]["image_size"]),
                                    float(cfg["TRAIN"].get("cam_margin", 1.3)))
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                lg_c = m.head(m.backbone.forward_head(m.backbone.forward_features(patch)))
            ok = (logits.argmax(1) == y).cpu().numpy()
            ok_c = (lg_c.argmax(1) == y).cpu().numpy()
            bx = boxes.cpu().numpy()
            for i in range(x.shape[0]):
                iid = int(ds.data_ids[seen + i])
                W, H = img_size(iid); px, py, pw, ph = gt_px[iid]
                g = (px / W, py / H, (px + pw) / W, (py + ph) / H)
                ix0 = max(bx[i, 0], g[0]); iy0 = max(bx[i, 1], g[1])
                ix1 = min(bx[i, 2], g[2]); iy1 = min(bx[i, 3], g[3])
                inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                area = (bx[i, 2] - bx[i, 0]) * (bx[i, 3] - bx[i, 1])
                garea = (g[2] - g[0]) * (g[3] - g[1])
                iou = inter / max(area + garea - inter, 1e-9)
                prec = inter / max(area, 1e-9)
                rec = inter / max(garea, 1e-9)
                rows.append((area, iou, prec, rec, ok[i], ok_c[i]))
                if a.fig and len(figs) < 8:
                    figs.append((x[i].cpu(), cam[i].cpu(), bx[i], g, ok[i], ok_c[i]))
            seen += x.shape[0]
            if seen >= a.n: break

    R = np.array(rows, dtype=float)
    print(f"样本 {len(R)}；框面积占比 mean={R[:,0].mean():.3f}  median={np.median(R[:,0]):.3f}")
    print(f"与 GT 框 IoU: mean={R[:,1].mean():.3f}  median={np.median(R[:,1]):.3f}")
    print(f"精确率(框内落在 GT 内) mean={R[:,2].mean():.3f}；召回率 mean={R[:,3].mean():.3f}")
    print(f"主分支 top-1 {R[:,4].mean()*100:.1f}%  裁剪分支 {R[:,5].mean()*100:.1f}%")
    q = np.quantile(R[:,1], [1/3, 2/3])
    for lo, hi, tag in ((0, q[0], "IoU 低"), (q[0], q[1], "中"), (q[1], 1.01, "高")):
        sel = (R[:,1] >= lo) & (R[:,1] < hi)
        if sel.sum():
            print(f"  {tag}三分位: n={int(sel.sum()):3d}  主分支 {R[sel,4].mean()*100:5.1f}%  "
                  f"裁剪 {R[sel,5].mean()*100:5.1f}%  面积 {R[sel,0].mean():.3f}")

    if a.fig and figs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        mean = np.array(dc["mean"], dtype=np.float32); std = np.array(dc["std"], dtype=np.float32)
        for ax, (img, cam, b, g, ok, okc) in zip(axes.ravel(), figs):
            im = img.permute(1, 2, 0).numpy() * std + mean      # 反归一化
            im = np.clip(im, 0, 1)
            H_, W_ = im.shape[:2]
            ax.imshow(im, extent=(0, W_, H_, 0))
            c = cam.numpy(); c = (c - c.min()) / (c.max() - c.min() + 1e-9)
            mask = np.ma.masked_where(c < 0.6, c)               # 只叠加最高响应的 40%
            # ⚠️ 必须给 extent：CAM 是 14×14，不给 extent 会把坐标轴重置成 0..14，
            # 把下面的照片挤掉（第一版可视化就是栽在这里）
            ax.imshow(mask, extent=(0, W_, H_, 0), alpha=0.5, cmap="jet")
            ax.set_xlim(0, W_); ax.set_ylim(H_, 0)
            for (x0, y0, x1, y1), col, lab in ((g, "red", "GT"), (b, "lime", "CAM")):
                ax.add_patch(plt.Rectangle((x0 * im.shape[1], y0 * im.shape[0]),
                                           (x1 - x0) * im.shape[1], (y1 - y0) * im.shape[0],
                                           fill=False, edgecolor=col, lw=2, label=lab))
            ax.set_title(f"main={'OK' if ok else 'X'} crop={'OK' if okc else 'X'}", fontsize=9)
            ax.axis("off")
        handles = [plt.Line2D([0], [0], color="red", lw=2, label="GT box"),
                   plt.Line2D([0], [0], color="lime", lw=2, label="CAM box")]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
        Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(a.fig, dpi=120)
        print("fig ->", a.fig)


if __name__ == "__main__":
    main()
