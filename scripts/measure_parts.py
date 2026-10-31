"""全数据集部件掩码测量：把"框变小了"从设计使然升级为可比较的证据。

英文审稿视角的三条硬要求（2026-10-31）：
  1. **同图同指标、多检查点并排**——单臂图无法支撑任何比较性结论；
  2. **全数据集分布**而不是挑 6 张——要中位/IQR/尾部，还要配对差；
  3. **有效部件数**（participation ratio）——K 个部件到底等于几个。

本脚本对每个 checkpoint 在**同一批固定抽样图**上算：
  · 框面积（14×14 格子数 / 196，以及占 448 输入的比例）
  · GT 鸟框富集倍数（框内质量 / 框面积占比）
  · 边界质量比（外两圈 / 均匀期望）
  · 框内填充率（80% 质量格 / 外接框格）
  · 两两 soft-IoU 均值、有效部件数 K_eff

K_eff 用**参与率**：(Σλ)² / Σλ²，λ 是 K×K 相似度矩阵的特征值。
K 个完全相同的部件 → K_eff = 1；两两正交 → K_eff = K。

用法：
    python scripts/measure_parts.py --checkpoint A.pth B.pth --labels 无紧致 紧致 --n 1200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.cub import CUBDataset
from dataset.transforms import build_transform, resolve_data_config
from engine.trainer import read_checkpoint, resolve_amp
from model.classifier import build_model, load_state_dict_tolerant
from utils.config import resolve_path


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


def measure(ckpt_path, idxs, dev, n_keep=200):
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
    crop_pct = float(data_cfg.get("crop_pct") or 1.0)
    ds = CUBDataset(data_root=root, split="test",
                    transform=build_transform(data_cfg, img_size, train=False),
                    num_classes=dcfg["num_classes"])
    gts, paths = load_gt(root)
    from PIL import Image
    amp = resolve_amp(cfg)

    rows = []
    dl = DataLoader(Subset(ds, idxs), batch_size=16, shuffle=False, num_workers=4)
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(dev)
            if amp:
                with torch.autocast(device_type="cuda", dtype=amp):
                    feat = model.backbone.forward_features(x)
                feat = feat.float()
            else:
                feat = model.backbone.forward_features(x)
            M = pa.part_masks(feat).float().cpu().numpy()      # [B,K,h,w]
            for bi in range(M.shape[0]):
                iid = ds.data_ids[idxs[len(rows)]]
                W, H = Image.open(root / "images" / paths[iid]).size
                s = int(round(img_size / crop_pct)) / min(W, H)
                nw, nh = (int(round(int(round(img_size / crop_pct)) * W / H)),
                          int(round(img_size / crop_pct))) if W >= H else \
                         (int(round(img_size / crop_pct)),
                          int(round(int(round(img_size / crop_pct)) * H / W)))
                ox, oy = (nw - img_size) / 2.0, (nh - img_size) / 2.0
                gx, gy, gw, gh = gts[iid]
                gx0, gy0 = gx * s - ox, gy * s - oy
                gbw, gbh = gw * s, gh * s
                m_all = M[bi]                                   # [K,h,w]
                K, h, w = m_all.shape
                yy, xx = np.mgrid[0:h, 0:w]
                inside = (((xx + 0.5) / w * img_size >= gx0) &
                          ((xx + 0.5) / w * img_size < gx0 + gbw) &
                          ((yy + 0.5) / h * img_size >= gy0) &
                          ((yy + 0.5) / h * img_size < gy0 + gbh))
                gt_area = max(inside.mean(), 1e-9)
                areas, ens, bord, fills = [], [], [], []
                for k in range(K):
                    m = m_all[k]
                    v = m.reshape(-1)
                    tot = max(float(v.sum()), 1e-12)
                    o = np.argsort(-v)
                    c = np.cumsum(v[o])
                    n80 = int(np.searchsorted(c, 0.8 * c[-1]) + 1)
                    ys, xs = np.divmod(o[:n80], w)
                    cells = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
                    areas.append(cells / (h * w))
                    fills.append(n80 / max(cells, 1))
                    ens.append(float(m[inside].sum() / tot) / gt_area)
                    ring = np.ones((h, w), bool); ring[2:h-2, 2:w-2] = False
                    bord.append(float(m[ring].sum() / tot) / max(ring.mean(), 1e-9))
                P = m_all / np.maximum(m_all.sum(axis=(1, 2), keepdims=True), 1e-12)
                flat = P.reshape(K, -1)
                flat = flat - flat.mean(1, keepdims=True)
                nrm = np.linalg.norm(flat, axis=1, keepdims=True).clip(1e-12)
                C = (flat / nrm) @ (flat / nrm).T
                ev = np.linalg.eigvalsh(C).clip(0, None)
                k_eff = float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-12))
                off = C[~np.eye(K, dtype=bool)]
                rows.append((np.mean(areas), np.median(ens), np.median(bord),
                             np.median(fills), off.mean(), k_eff))
            if len(rows) >= n_keep:
                break

    a = np.array(rows)
    return dict(area=a[:, 0], gt=a[:, 1], border=a[:, 2], fill=a[:, 3],
                iou=a[:, 4], k_eff=a[:, 5], centroid=a[:, 6])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--n", type=int, default=1200, help="等距抽样的图片数（全 test 为 5794）")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = a.labels or [Path(p).parent.parent.name for p in a.checkpoint]
    idxs = np.linspace(0, 5793, a.n).astype(int).tolist()

    res = {}
    for ckp, lab in zip(a.checkpoint, labels):
        res[lab] = measure(ckp, idxs, dev)
        print(f"[完成] {lab}", flush=True)

    keys = [("area", "框面积（占 14×14 网格）", 100),
            ("gt", "GT 富集倍数", 1),
            ("border", "边界质量比", 1),
            ("fill", "框内填充率", 100),
            ("iou", "部件两两余弦", 1),
            ("k_eff", "有效部件数 K_eff", 1),
            ("centroid", "质心两两距离（14 格制）", 1)]
    print(f"\n抽样 n={a.n}（等距，全 test 5794）")
    hdr = f"{'指标':26s}" + "".join(f"{l:>16s}" for l in labels)
    print(hdr); print("-" * len(hdr))
    for k, name, sc in keys:
        line = f"{name:26s}"
        for lab in labels:
            v = res[lab][k] * sc
            line += f"{np.median(v):9.2f}±{np.percentile(v,75)-np.percentile(v,25):5.2f}"
        print(line)
    print("\n格式：中位 ± IQR")


if __name__ == "__main__":
    main()
