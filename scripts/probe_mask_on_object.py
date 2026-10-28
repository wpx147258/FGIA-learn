"""定量：部件注意力/掩码有多少质量落在鸟（GT bbox）上。

TL 的权重图或 PartAttention 的掩码都被归一化成每个部件总和 1，然后在 GT 框内求和。
  * 均匀注意力：质量占比 ≈ GT 框面积占比；
  * 部件在看鸟：占比显著高于面积占比；
  * 部件在看背景：占比低于面积占比。

用法：python scripts/probe_mask_on_object.py --checkpoint <ckpt> [--n 96]
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
    cfg = ck["config"]; amp = resolve_amp(cfg)
    m = build_model(cfg).to(dev); m.load_state_dict(ck["model"]); m.eval()
    root = resolve_path(cfg["DATASET"]["root"])
    # GT 框（归一化）：需要原图尺寸
    with open(root / "images.txt") as f:
        paths = {int(l.split()[0]): l.split()[1] for l in f if l.strip()}
    with open(root / "bounding_boxes.txt") as f:
        gt_px = {int(l.split()[0]): tuple(map(float, l.split()[1:])) for l in f if len(l.split()) == 5}
    from PIL import Image
    # GT 框在**原图**坐标里，而网络看到的是「短边缩放到 resize_size，再中心裁 image_size」。
    # 之前直接把原图归一化坐标按 h×h 网格切，忽略了这层平移/缩放：短边会被裁掉
    # (resize_size − image_size) 个像素，框靠边时误差不小，而富集比是判据本身。
    # 这里把框过一遍同样的几何变换，再归一化到网络输入。
    data_cfg = resolve_data_config(m.backbone)
    crop_pct = data_cfg.get("crop_pct") or 1.0
    img_size = 448
    resize_size = int(round(img_size / crop_pct))
    cache = {}
    def gbox(i):
        if i not in cache:
            W, H = Image.open(root / "images" / paths[i]).size
            s = resize_size / min(W, H)
            if W >= H:
                nw, nh = int(resize_size * W / H), resize_size
            else:
                nw, nh = resize_size, int(resize_size * H / W)
            off_x, off_y = (nw - img_size) / 2.0, (nh - img_size) / 2.0
            x, y, w, h = gt_px[i]
            x0 = min(max((x * s - off_x) / img_size, 0.0), 1.0)
            y0 = min(max((y * s - off_y) / img_size, 0.0), 1.0)
            x1 = min(max(((x + w) * s - off_x) / img_size, 0.0), 1.0)
            y1 = min(max(((y + h) * s - off_y) / img_size, 0.0), 1.0)
            cache[i] = (x0, y0, x1, y1)
        return cache[i]

    ds = CUBDataset(root, "test", build_transform(resolve_data_config(m.backbone), 448, train=False), 200)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=cfg["DATASET"]["num_workers"])
    tl = m.tokenlearner; pa = getattr(m.backbone, "part_attn", None)
    inside, areas, seen = [], [], 0
    for batch in dl:
        x = batch["image"].to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            f = m.backbone.forward_features(x); fm = m._to_feature_map(f).float()
            if tl is not None:
                t = tl.proj(fm).flatten(2).transpose(1, 2)
                w = tl.score(t).softmax(dim=1).transpose(1, 2)          # [B,K,N]
            else:
                w = pa.part_masks(fm).flatten(2)                        # [B,K,N]
                w = w / w.sum(-1, keepdim=True).clamp_min(1e-6)
        h = int(w.shape[-1] ** 0.5)
        w = w.reshape(w.shape[0], w.shape[1], h, h).cpu().numpy()
        for i in range(x.shape[0]):
            g = gbox(int(ds.data_ids[seen + i]))
            y0 = int(g[1] * h); y1 = max(y0 + 1, int(np.ceil(g[3] * h)))
            x0 = int(g[0] * h); x1 = max(x0 + 1, int(np.ceil(g[2] * h)))
            for k in range(w.shape[1]):
                mk = w[i, k]; tot = mk.sum()
                if tot <= 0: continue
                inside.append(mk[y0:y1, x0:x1].sum() / tot)
                areas.append((g[2] - g[0]) * (g[3] - g[1]))
        seen += x.shape[0]
        if seen >= a.n: break
    inside = np.array(inside); areas = np.array(areas)
    print(f"样本 {seen}，部件 {len(inside)//max(seen,1)}/图 × {seen}")
    print(f"每部件落在 GT 框内的质量占比: mean={inside.mean():.3f}  median={np.median(inside):.3f}")
    print(f"GT 框面积占比（均匀注意力的期望）: mean={areas.mean():.3f}")
    print(f"富集比（质量占比 / 面积占比）: {inside.mean()/areas.mean():.2f}×")


if __name__ == "__main__":
    main()
