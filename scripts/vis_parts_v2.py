"""部件可视化 v2（2026-10-31 重写，依据三份审查意见）。

相对 v1 的四项修正——都是**测量口径**问题，不只是画得好看：

1. **坐标系**：v1 把 14×14 掩码上采样到 448 后直接盖在**原图**上，但掩码活在
   448 中心裁剪坐标系里。448 裁剪只覆盖原图约 54%（中位）的面积，所以 v1 里
   "框贴到图像边缘"其实是"贴到裁剪窗口边缘"——实测 96 个框里 **0 个**贴原图边，
   **64 个**贴裁剪窗口边。本版直接显示**网络输入**（448 裁剪），掩码与图像逐像素对齐，
   零映射误差；原图只在需要时另行给出。
2. **框的定义**：v1 画的是"累计 80% 质量"的外接框，而 `part_rerank` 实际裁的是
   `peak_mask(alpha) → mask_to_box → margin` 外扩，两者面积差 1.7 倍（25.5% vs 43.8%
   的 448 输入）。本版两框都给，且注明。
3. **统计口径**：v1 打印的 `ln(K·h·w)` 熵上限与"均匀=100%"都是错的。
   逐部件归一化后熵上限是 ln(h·w)=5.28；均匀分布时承载 80% 质量需要 80% 的格子，
   **不是 100%**。本版改正，并补上真正该看的四个量：两两 IoU、边界质量比、
   GT 富集倍数、框内填充率。
4. **叠加方式**：v1 是 `imshow(img, alpha=.35) + imshow(mask, alpha=.6)`，
   合成后原图只剩 14% 不透明度，鸟基本看不见。本版原图 100% 不透明，
   掩码用逐像素 alpha 的 RGBA（magma），只在有响应处着色。

用法：
    python scripts/vis_parts_v2.py --checkpoint <ckpt> --n 8 --out figs/xxx.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.cub import CUBDataset
from dataset.transforms import build_transform, resolve_data_config
from engine.trainer import read_checkpoint, resolve_amp
from model.classifier import build_model, load_state_dict_tolerant
from utils.config import resolve_path


def load_gt_boxes(root: Path) -> dict[int, tuple[float, float, float, float]]:
    """CUB 的 GT 鸟框（原图像素坐标，x y w h）。"""
    out = {}
    with open(root / "bounding_boxes.txt") as f:
        for line in f:
            iid, x, y, w, h = line.split()
            out[int(iid)] = (float(x), float(y), float(w), float(h))
    return out


def geom(W: int, H: int, img_size: int, crop_pct: float):
    """原图 → 网络输入 的几何参数：缩放 s 与中心裁剪偏移 (ox, oy)。"""
    resize = int(round(img_size / crop_pct))
    s = resize / min(W, H)
    nw, nh = (int(round(resize * W / H)), resize) if W >= H else \
             (resize, int(round(resize * H / W)))
    return s, (nw - img_size) / 2.0, (nh - img_size) / 2.0


def part_metrics(m: np.ndarray, gt_inside: np.ndarray | None, border: int = 2):
    """单个部件掩码 [h,w] 的四项指标。"""
    h, w = m.shape
    v = m.reshape(-1)
    tot = max(float(v.sum()), 1e-12)
    p = v / tot
    ent = float(-(p * np.log(p + 1e-12)).sum())          # 上限 ln(h*w)
    order = np.argsort(-v)
    cum = np.cumsum(v[order])
    n80 = int(np.searchsorted(cum, 0.8 * cum[-1]) + 1)
    ring = np.ones((h, w), bool)
    ring[border:h - border, border:w - border] = False
    border_ratio = float(m[ring].sum() / tot) / max(ring.mean(), 1e-9)
    ys, xs = np.divmod(order[:n80], w)
    bbox_cells = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    fill = n80 / max(bbox_cells, 1)
    gt_enrich = None
    if gt_inside is not None and gt_inside.any():
        gt_enrich = float(m[gt_inside].sum() / tot) / max(gt_inside.mean(), 1e-9)
    return dict(ent=ent, f80=n80 / (h * w), border=border_ratio,
                fill=fill, gt=gt_enrich,
                box=(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cols", type=int, default=8, help="每行几个样本组")
    ap.add_argument("--out", default="figs/parts_v2.png")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = read_checkpoint(a.checkpoint, map_location=dev)
    cfg = ck["config"]
    model = build_model(cfg).to(dev)
    load_state_dict_tolerant(model, ck["model"])
    model.eval()
    pa = getattr(model.backbone, "part_attn", None)
    if pa is None:
        sys.exit("该 checkpoint 没有 part_attn")

    dcfg = cfg["DATASET"]
    img_size = int(dcfg["image_size"])
    root = resolve_path(dcfg["root"])
    data_cfg = dict(resolve_data_config(model.backbone))
    if dcfg.get("crop_pct"):
        data_cfg["crop_pct"] = float(dcfg["crop_pct"])
    crop_pct = float(data_cfg.get("crop_pct") or 1.0)
    mean = np.array(data_cfg["mean"], np.float32).reshape(3, 1, 1)
    std = np.array(data_cfg["std"], np.float32).reshape(3, 1, 1)

    ds = CUBDataset(data_root=root, split="test",
                    transform=build_transform(data_cfg, img_size, train=False),
                    num_classes=dcfg["num_classes"])
    gts = load_gt_boxes(root)
    idxs = np.linspace(0, len(ds) - 1, a.n).astype(int).tolist()
    batch = next(iter(DataLoader(Subset(ds, idxs), batch_size=a.n, shuffle=False, num_workers=2)))

    amp = resolve_amp(cfg)
    with torch.no_grad():
        x = batch["image"].to(dev)
        feat = model.backbone.forward_features(x)
        masks = pa.part_masks(feat.float()).float().cpu().numpy()   # [B,K,h,w]
    B, K, h, w = masks.shape
    imgs = (batch["image"].numpy() * std + mean).clip(0, 1).transpose(0, 2, 3, 1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib import patheffects as pe
    from matplotlib.patches import Rectangle

    PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
               "#0072B2", "#D55E00", "#CC79A7", "#000000"][:K]

    # 每个样本占 (1 + K) 列：网络输入 + K 个部件
    cw = 1 + K
    fig, axes = plt.subplots(B, cw, figsize=(1.55 * cw, 1.9 * B))
    axes = np.atleast_2d(axes)
    halo = [pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()]

    print(f"K={K}  网格 {h}x{w}  熵上限 ln(h*w)={np.log(h*w):.2f}  "
          f"(均匀分布时 f80 = {0.8*100:.0f}%，不是 100%)")
    allm = []

    for bi in range(B):
        iid = ds.data_ids[idxs[bi]]
        fname = ds.imgs_paths[iid]
        W_img, H_img = plt.imread(fname).shape[1], plt.imread(fname).shape[0]
        s, ox, oy = geom(W_img, H_img, img_size, crop_pct)
        gx, gy, gw, gh = gts[iid]
        # GT 框 → 448 输入坐标
        gx0, gy0 = gx * s - ox, gy * s - oy
        gbw, gbh = gw * s, gh * s

        yy, xx = np.mgrid[0:h, 0:w]
        xc = (xx + 0.5) / w * img_size
        yc = (yy + 0.5) / h * img_size
        gt_inside = ((xc >= gx0) & (xc < gx0 + gbw) & (yc >= gy0) & (yc < gy0 + gbh))

        ax0 = axes[bi][0]
        ax0.imshow(imgs[bi])
        ax0.add_patch(Rectangle((gx0, gy0), gbw, gbh, fill=False, ec="lime",
                                lw=1.4, ls="--", path_effects=halo, zorder=6))
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.set_title(f"{Path(str(fname)).parent.name[:22]}\n448 crop + GT (green)", fontsize=5.5)

        # 两两 soft-IoU（逐部件归一化后的空间分布）
        P = masks[bi] / np.maximum(masks[bi].sum(axis=(1, 2), keepdims=True), 1e-12)
        inter = np.minimum(P[:, None], P[None, :]).sum(axis=(2, 3))
        union = np.maximum(P[:, None], P[None, :]).sum(axis=(2, 3))
        iou = inter / np.maximum(union, 1e-12)
        np.fill_diagonal(iou, 0.0)

        for k in range(K):
            m = masks[bi, k]
            met = part_metrics(m, gt_inside)
            allm.append(met)
            up = F.interpolate(torch.from_numpy(m)[None, None], size=(img_size, img_size),
                               mode="bilinear", align_corners=False)[0, 0].numpy()
            ax = axes[bi][k + 1]
            ax.imshow(imgs[bi])                       # 原图 100% 不透明
            lo, hi = float(np.percentile(up, 5)), float(np.percentile(up, 99.5))
            nrm = np.clip((up - lo) / max(hi - lo, 1e-9), 0, 1) ** 0.8
            rgba = matplotlib.colormaps["magma"](nrm)
            rgba[..., 3] = 0.85 * nrm                  # 无响应处完全透明
            ax.imshow(rgba, interpolation="nearest")   # 不额外平滑 14x14
            x0, y0, x1, y1 = met["box"]
            ax.add_patch(Rectangle((x0 / w * img_size, y0 / h * img_size),
                                   (x1 - x0) / w * img_size, (y1 - y0) / h * img_size,
                                   fill=False, ec="white", lw=0.9, ls=":", zorder=6))
            ax.set_xticks([]); ax.set_yticks([])
            txt = (f"P{k+1}  H={met['ent']:.2f}\n"
                   f"80%跑{met['f80']*100:.0f}%格 填充{met['fill']*100:.0f}%\n"
                   f"边{met['border']:.2f}x  GT{met['gt']:.2f}x" if met["gt"] else
                   f"P{k+1}  H={met['ent']:.2f}")
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
                    fontsize=4.6, color="w",
                    bbox=dict(boxstyle="round,pad=0.15", fc="k", ec="none", alpha=0.45))

    handles = [Line2D([0], [0], color="lime", lw=1.4, ls="--", label="GT bird box"),
               Line2D([0], [0], color="white", lw=0.9, ls=":", label="80% mass box")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=6)
    fig.suptitle(f"{Path(a.checkpoint).parent.parent.name} · part_masks over the 448 network input "
                 f"(masks are on a {h}x{w} grid)", fontsize=7)
    plt.tight_layout(rect=(0, 0.02, 1, 0.98))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"已保存 {a.out}")

    ent = np.array([m["ent"] for m in allm])
    f80 = np.array([m["f80"] for m in allm])
    brd = np.array([m["border"] for m in allm])
    fil = np.array([m["fill"] for m in allm])
    gt = np.array([m["gt"] for m in allm if m["gt"] is not None])
    print(f"熵      中位 {np.median(ent):.2f} / 上限 {np.log(h*w):.2f}")
    print(f"f80(承载 80% 质量所需格子/全网格) 中位 {np.median(f80)*100:.1f}%   "
          f"[均匀 = {0.8*100:.0f}%]")
    print(f"边界质量比（外 {2} 圈 / 均匀期望） 中位 {np.median(brd):.2f}x   [1.0 = 与均匀相同]")
    print(f"框内填充率（80% 质量格 / 外接框格） 中位 {np.median(fil)*100:.0f}%   [低 = 散斑非成片]")
    if len(gt):
        print(f"GT 富集倍数（框内质量 / 框面积占比） 中位 {np.median(gt):.2f}x   [>1 = 偏向鸟]")


if __name__ == "__main__":
    main()
