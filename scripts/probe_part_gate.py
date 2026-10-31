"""零训练诊断：给部件掩码乘一个「自生成定位门」，看能否把质量赶到鸟身上。

背景（2026-10-31）：
  · 全量测量（measure_parts.py）显示部件掩码的 GT 富集只有 0.98~1.15×，边界比 1.42×
    ——质量既不在鸟上、还偏向裁剪窗口边缘；
  · 同一批图上 CAM 裁剪框面积 12%、框内 GT 覆盖率 98%——**同一个模型里，CAM 定位得很准，
    部件的自由学习掩码完全不定位**；
  · 因果干预（probe_part_causal2.py）证明掩码内容对预测没有影响。

所以问题不是"掩码不够紧"，而是"掩码不知道往哪看"。这个探针检验：**乘上一个不依赖 GT 的
定位门，能不能把掩码质量的 GT 富集从 ~1.0 提到 1.3 以上。**

三种门（都不使用 GT，只用模型自己的特征/权重）：
  energy   ||feat||_c，逐图 robust 归一化后 sigmoid —— 最便宜，只用特征幅度
  cam      分类头权重 × 特征图（预测类），与训练时的 CAM 裁剪分支同一套计算
  center   高斯中心先验 —— 纯对照，判断"只要往中间收"是不是就够了

用法：
    python scripts/probe_part_gate.py --checkpoint <ckpt> --n 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.cub import CUBDataset
from dataset.transforms import build_transform, resolve_data_config
from engine.trainer import read_checkpoint, resolve_amp
from model.cam_aug import cam_map
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


def metrics(m_all, inside, border=2):
    """返回 (GT 富集, 边界比, 80% 质量框面积, 两两余弦)。"""
    K, h, w = m_all.shape
    gt_area = max(inside.mean(), 1e-9)
    ring = np.ones((h, w), bool); ring[border:h-border, border:w-border] = False
    ents, areas, ens, brs = [], [], [], []
    P = []
    for k in range(K):
        m = m_all[k]
        tot = max(float(m.sum()), 1e-12)
        ens.append(float(m[inside].sum() / tot) / gt_area)
        brs.append(float(m[ring].sum() / tot) / max(ring.mean(), 1e-9))
        v = m.reshape(-1); o = np.argsort(-v); c = np.cumsum(v[o])
        n = int(np.searchsorted(c, 0.8 * c[-1]) + 1)
        ys, xs = np.divmod(o[:n], w)
        areas.append(((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) / (h * w))
        P.append(m / tot)
    P = np.array(P)
    Pf = P.reshape(K, -1)
    Pf = Pf - Pf.mean(1, keepdims=True)
    Pf = Pf / np.linalg.norm(Pf, axis=1, keepdims=True).clip(1e-12)
    C = Pf @ Pf.T
    ev = np.linalg.eigvalsh(C).clip(0, None)
    k_eff = float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-12))
    return np.median(ens), np.median(brs), np.mean(areas), np.median(C[~np.eye(K, dtype=bool)]), k_eff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=400)
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
    ds = CUBDataset(data_root=root, split="test",
                    transform=build_transform(data_cfg, img_size, train=False),
                    num_classes=dcfg["num_classes"])
    gts, paths = load_gt(root)
    idxs = np.linspace(0, 5793, a.n).astype(int).tolist()
    amp = resolve_amp(cfg)

    res = {k: [] for k in ("none", "energy", "cam", "center")}
    dl = DataLoader(Subset(ds, idxs), batch_size=16, shuffle=False, num_workers=4)
    seen = 0
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(dev)
            if amp:
                with torch.autocast(device_type="cuda", dtype=amp):
                    feat = model.backbone.forward_features(x)
                feat = feat.float()
            else:
                feat = model.backbone.forward_features(x)
            masks = pa.part_masks(feat).float()                    # [B,K,h,w]
            logits = model.head(model.backbone.forward_head(feat))
            pred = logits.argmax(1)
            w_head = model.head.cam_weight(feat.shape[1]).float()
            cam = cam_map(feat, w_head, pred)                      # [B,h,w]

            B, K, h, w = masks.shape
            e = feat.norm(dim=1)                                   # [B,h,w]
            ef = e.flatten(2)
            med = ef.median(-1).values.median(-1).values.view(B, 1, 1)
            q75 = ef.quantile(0.75, -1).quantile(0.75, -1).view(B, 1, 1)
            q25 = ef.quantile(0.25, -1).quantile(0.25, -1).view(B, 1, 1)
            iqr = q75 - q25
            g_energy = torch.sigmoid((e - med) / (iqr + 1e-6) * 2.0).unsqueeze(1)

            cf = cam.flatten(1)
            cmax = cf.max(1).values.view(B, 1, 1)
            g_cam = (cam / cmax.clamp_min(1e-9)).unsqueeze(1)      # 0~1
            g_cam = (g_cam > 0.5).float() * 0.75 + 0.25            # 支撑区=1，其余=0.25（软门）

            yy = torch.linspace(-1, 1, h, device=dev).view(1, 1, h, 1)
            xx = torch.linspace(-1, 1, w, device=dev).view(1, 1, 1, w)
            g_center = torch.exp(-(yy ** 2 + xx ** 2) / 0.8)

            for name, g in (("none", None), ("energy", g_energy), ("cam", g_cam), ("center", g_center)):
                mm = masks if g is None else masks * g
                for bi in range(B):
                    iid = ds.data_ids[idxs[seen + bi]]
                    W, H = Image.open(root / "images" / paths[iid]).size
                    rr = int(round(img_size / crop_pct)); s = rr / min(W, H)
                    nw, nh = ((int(round(rr * W / H)), rr) if W >= H else (rr, int(round(rr * H / W))))
                    ox, oy = (nw - img_size) / 2.0, (nh - img_size) / 2.0
                    gx, gy, gw, gh = gts[iid]
                    gx0, gy0 = gx * s - ox, gy * s - oy
                    gbw, gbh = gw * s, gh * s
                    yy2, xx2 = np.mgrid[0:h, 0:w]
                    inside = (((xx2 + 0.5) / w * img_size >= gx0) & ((xx2 + 0.5) / w * img_size < gx0 + gbw) &
                              ((yy2 + 0.5) / h * img_size >= gy0) & ((yy2 + 0.5) / h * img_size < gy0 + gbh))
                    res[name].append(metrics(mm[bi].cpu().numpy(), inside))
            seen += B
            if seen >= a.n:
                break

    print(f"n={min(seen, a.n)}  门控对部件掩码的影响（中位）\n")
    print(f"{'门':10s}{'GT富集':>9s}{'边界比':>9s}{'框面积':>9s}{'两两余弦':>10s}{'K_eff':>8s}")
    for name in ("none", "energy", "cam", "center"):
        arr = np.array(res[name])
        print(f"{name:10s}{np.median(arr[:,0]):9.2f}{np.median(arr[:,1]):9.2f}"
              f"{np.median(arr[:,2])*100:8.1f}%{np.median(arr[:,3]):10.2f}{np.median(arr[:,4]):8.2f}")


if __name__ == "__main__":
    main()
