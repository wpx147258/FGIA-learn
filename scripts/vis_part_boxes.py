"""可视化：把 PartAttention 每个部件实际会裁的区域画回原图。

为什么要这个：`part_rerank` 要在测试时按部件掩码裁区域再重编码，它的成败完全取决于
裁出来的框是不是"一只鸟的局部"。历史上（5,395 口径、碎斑掩码）实测每个部件的框
覆盖全图 **98.5%**，裁出来还是整图，所以那条路当时必然是无效的。
现在掩码可以被压成真部件（重叠 0.009），这个脚本用来直接看框对不对。

框的取法：按掩码值从大到小累加，取到累计质量 ≥ --mass（默认 0.8）的那些格子，
再取它们的外接矩形——即"承担了 80% 质量的区域"，比阈值法稳。

坐标映射要反着做一遍 probe_mask_on_object.py 那套：
  原图 → 短边缩放到 resize_size → 中心裁 image_size → 归一化
所以反向是 x = (u·image_size + off_x) / s。

用法：
    python scripts/vis_part_boxes.py --checkpoint outputs/r50_all_pa4k8_s1/<ts>/swa_f0.5.pth \
        --n 6 --out figs/part_boxes.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.cub import CUBDataset
from dataset.transforms import build_transform, resolve_data_config
from engine.trainer import read_checkpoint, resolve_amp
from model.classifier import build_model, load_state_dict_tolerant
from utils.config import resolve_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=6, help="画几张图")
    ap.add_argument("--mass", type=float, default=0.8, help="框内要包含的掩码质量比例")
    ap.add_argument("--out", default="figs/part_boxes.png")
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
    crop_pct = data_cfg.get("crop_pct") or 1.0
    resize_size = int(round(img_size / crop_pct))

    ds = CUBDataset(data_root=root, split="test",
                    transform=build_transform(data_cfg, img_size, train=False),
                    num_classes=dcfg["num_classes"])
    dl = DataLoader(ds, batch_size=a.n, shuffle=False, num_workers=2)

    batch = next(iter(dl))
    x = batch["image"].to(dev)
    amp_dtype = resolve_amp(cfg)
    with torch.no_grad():
        if amp_dtype:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                feat = model.backbone.forward_features(x)
            feat = feat.float()
        else:
            feat = model.backbone.forward_features(x)
        masks = pa.part_masks(feat)                     # [B,K,h,w]
    masks = masks.float().cpu().numpy()
    B, K, h, w = masks.shape

    # 反变换：归一化坐标 (u,v) ∈ [0,1]（在裁剪后的图里）→ 原图像素
    def to_orig(u, v, W, H):
        s = resize_size / min(W, H)
        nw, nh = (int(resize_size * W / H), resize_size) if W >= H else \
                 (resize_size, int(resize_size * H / W))
        off_x, off_y = (nw - img_size) / 2.0, (nh - img_size) / 2.0
        return (u * img_size + off_x) / s, (v * img_size + off_y) / s

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:K]
    fig, axes = plt.subplots(2, (a.n + 1) // 2, figsize=(4.2 * ((a.n + 1) // 2), 8.6))
    axes = np.atleast_2d(axes)
    areas_all = []

    for bi in range(B):
        idx = bi
        rel = ds.data_ids[idx] if hasattr(ds, "data_ids") else None
        fname = ds.imgs_paths[ds.data_ids[idx]]
        img = Image.open(root / "images" / fname).convert("RGB") if fname else None
        ax = axes[bi // ((a.n + 1) // 2)][bi % ((a.n + 1) // 2)]
        if img is None:
            ax.axis("off"); continue
        W, H = img.size
        ax.imshow(img)
        for k in range(K):
            m = masks[bi, k].reshape(-1)
            order = np.argsort(-m)
            cum = np.cumsum(m[order])
            n_need = int(np.searchsorted(cum, a.mass * cum[-1]) + 1)
            sel = order[:n_need]
            ys, xs = np.divmod(sel, w)
            v0, v1 = ys.min() / h, (ys.max() + 1) / h
            u0, u1 = xs.min() / w, (xs.max() + 1) / w
            x0, y0 = to_orig(u0, v0, W, H)
            x1, y1 = to_orig(u1, v1, W, H)
            area = (x1 - x0) * (y1 - y0) / (W * H)
            areas_all.append(area)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor=colors[k], lw=2))
            ax.text(x0, max(y0 - 4, 8), f"P{k+1} {area*100:.0f}%", color=colors[k],
                    fontsize=8, weight="bold")
        ax.set_title(f"{fname}", fontsize=8)
        ax.axis("off")

    for j in range(B, axes.size):
        axes.flat[j].axis("off")
    plt.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(a.out, dpi=130)
    print(f"已保存 {a.out}")
    areas = np.array(areas_all) * 100
    print(f"框面积占原图: 中位 {np.median(areas):.1f}%  "
          f"25/75 分位 {np.percentile(areas,25):.1f}% / {np.percentile(areas,75):.1f}%  "
          f"最大 {areas.max():.1f}%")
    print("对照：历史上碎斑掩码的部件框覆盖全图 98.5%；整图 = 100%")


if __name__ == "__main__":
    main()
