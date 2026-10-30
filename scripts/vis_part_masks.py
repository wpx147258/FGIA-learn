"""可视化每个部件的**特征/掩码图本身**（不只是外接框）。

为什么需要：外接框只说明"范围有多大"，看不出"这一片是成块的还是散点"。
掩码图能直接回答：
  - 每个部件是集中在鸟身上的一块，还是铺满全图的噪声？
  - 不同部件之间到底像不像（对应 pa_overlap）？
  - 有没有哪个部件明显"学会了"（少数部件有效、其余躺平是常见病）。

每行 = 一张图：左边是原图，右边 K 张是各部件掩码（14×14 上采样，固定色标），
掩码上还画出该部件「累计 80% 质量」的外接框，便于同时看范围与形状。

用法：
    python scripts/vis_part_masks.py --checkpoint outputs/r50_all_pa4k8_s1/<ts>/swa_f0.5.pth \
        --n 12 --out figs/part_masks.png
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
from model.classifier import build_model, load_state_dict_tolerant
from utils.config import resolve_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="figs/part_masks.png")
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

    ds = CUBDataset(data_root=root, split="test",
                    transform=build_transform(data_cfg, img_size, train=False),
                    num_classes=dcfg["num_classes"])
    idxs = np.linspace(0, len(ds) - 1, a.n).astype(int).tolist()
    dl = DataLoader(Subset(ds, idxs), batch_size=a.n, shuffle=False, num_workers=2)
    batch = next(iter(dl))

    amp_dtype = resolve_amp(cfg)
    with torch.no_grad():
        if amp_dtype:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                feat = model.backbone.forward_features(batch["image"].to(dev))
            feat = feat.float()
        else:
            feat = model.backbone.forward_features(batch["image"].to(dev))
        masks = pa.part_masks(feat).float().cpu()      # [B,K,h,w]
    B, K, h, w = masks.shape

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(B, K + 1, figsize=(1.75 * (K + 1), 1.75 * B))
    axes = np.atleast_2d(axes)

    stats = []
    for bi in range(B):
        idx = idxs[bi]
        fname = ds.imgs_paths[ds.data_ids[idx]]
        img = Image.open(root / "images" / fname).convert("RGB")
        ax0 = axes[bi][0]
        ax0.imshow(img); ax0.set_xticks([]); ax0.set_yticks([])
        ax0.set_ylabel(Path(str(fname)).name.split("_")[-2] if "_" in str(fname) else "",
                       fontsize=6)
        for k in range(K):
            m = masks[bi, k].numpy()
            up = F.interpolate(torch.from_numpy(m)[None, None], size=(img_size, img_size),
                               mode="bilinear", align_corners=False)[0, 0].numpy()
            ax = axes[bi][k + 1]
            ax.imshow(img, alpha=0.35)
            ax.imshow(up, cmap="jet", alpha=0.6, vmin=0, vmax=float(masks[bi].max()))
            v = m.reshape(-1); o = np.argsort(-v); c = np.cumsum(v[o])
            n = int(np.searchsorted(c, 0.8 * c[-1]) + 1)
            ys, xs = np.divmod(o[:n], w)
            ax.add_patch(plt.Rectangle((xs.min() / w * img_size, ys.min() / h * img_size),
                                       (xs.max() - xs.min() + 1) / w * img_size,
                                       (ys.max() - ys.min() + 1) / h * img_size,
                                       fill=False, edgecolor="white", lw=1))
            ax.set_xticks([]); ax.set_yticks([])
            if bi == 0:
                ax.set_title(f"P{k+1}", fontsize=7)
            # 统计：熵、以及"峰值格子占比"
            p = v / v.sum()
            ent = float(-(p * np.log(p + 1e-12)).sum())
            frac80 = n / (h * w)
            stats.append((ent, frac80))
    plt.tight_layout(pad=0.15)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(a.out, dpi=125)
    print(f"已保存 {a.out}")

    st = np.array(stats)
    print(f"部件掩码熵: 中位 {np.median(st[:,0]):.2f} / 上限 ln{K*h*w}={np.log(K*h*w):.2f}")
    print(f"承载 80% 质量所需的格子占全图比例: 中位 {np.median(st[:,1])*100:.0f}% "
          f"(100% = 完全均匀)")
    print("注：熵用的是整幅 K*h*w 上的分布做参照，逐部件归一化更准确的值见训练日志 pa_entropy")


if __name__ == "__main__":
    main()
