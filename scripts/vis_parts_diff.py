"""部件**分化**专用图（2026-10-31，依据审查意见）。

三张热力图并排只能靠眼睛比；回答"8 个部件到底分不分化"最短的路径是三个面板：

  ① 共识方差图：std_k p_k —— 全零 = 8 个部件完全一样（塌缩），有结构 = 有分工
  ② argmax + 主导度：每个位置归哪个部件、多确定
  ③ K×K soft-IoU 矩阵：两两有多像（这是 `pa_dup` 罚项直接对应的量）

同一张图里放多个 checkpoint 做对照，差异会非常直观：
  compact（紧致版）mean IoU ≈ 0.01；ovlstr（允许重叠版）mean IoU ≈ 0.86。

用法：
    python scripts/vis_parts_diff.py --checkpoint A.pth B.pth --labels 紧致 重叠 --out figs/diff.png
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


def get_masks(ckpt_path, idxs, dev):
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
    batch = next(iter(DataLoader(Subset(ds, idxs), batch_size=len(idxs), shuffle=False, num_workers=2)))
    with torch.no_grad():
        feat = model.backbone.forward_features(batch["image"].to(dev))
        masks = pa.part_masks(feat.float()).float().cpu().numpy()
    mean = np.array(data_cfg["mean"], np.float32).reshape(3, 1, 1)
    std = np.array(data_cfg["std"], np.float32).reshape(3, 1, 1)
    imgs = (batch["image"].numpy() * std + mean).clip(0, 1).transpose(0, 2, 3, 1)
    return masks, imgs, Path(ckpt_path).parent.parent.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--sample", type=int, default=0, help="用等距取样里的第几个样本")
    ap.add_argument("--out", default="figs/parts_diff.png")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = a.labels or [Path(p).parent.parent.name for p in a.checkpoint]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idxs = np.linspace(0, 5793, 6).astype(int).tolist()
    fig, axes = plt.subplots(len(a.checkpoint), 3, figsize=(11.4, 3.5 * len(a.checkpoint)))
    axes = np.atleast_2d(axes)

    for r, (ckp, lab) in enumerate(zip(a.checkpoint, labels)):
        masks, imgs, _ = get_masks(ckp, idxs, dev)
        M = masks[a.sample]                                   # [K,h,w]
        K, h, w = M.shape
        P = M / np.maximum(M.sum(axis=(1, 2), keepdims=True), 1e-12)   # 逐部件空间分布

        # ① 共识方差
        var = P.std(axis=0)
        ax = axes[r][0]
        ax.imshow(imgs[a.sample])
        vmax = max(float(var.max()), 1e-9)
        ax.imshow(var / vmax, cmap="magma", alpha=0.85)
        ax.set_title(f"[{lab}] std across parts  max={vmax:.3f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        # ② argmax 部件 + 主导度
        arg = P.argmax(axis=0)
        top2 = np.partition(P, -2, axis=0)[-2]
        marg = P.max(axis=0) - top2
        marg = marg / max(float(marg.max()), 1e-9)
        cmap = plt.get_cmap("tab10").resampled(K)
        rgba = cmap(arg)
        rgba[..., 3] = 0.9 * marg
        ax = axes[r][1]
        ax.imshow(imgs[a.sample])
        ax.imshow(rgba, interpolation="nearest", alpha=0.85)
        ax.set_title(f"[{lab}] argmax part (alpha = dominance)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        # ③ K×K soft-IoU
        inter = np.minimum(P[:, None], P[None, :]).sum(axis=(2, 3))
        union = np.maximum(P[:, None], P[None, :]).sum(axis=(2, 3))
        iou = inter / np.maximum(union, 1e-12)
        off = iou[~np.eye(K, dtype=bool)]
        ax = axes[r][2]
        im = ax.imshow(iou, cmap="magma", vmin=0, vmax=1)
        for i in range(K):
            for j in range(K):
                if i != j:
                    ax.text(j, i, f"{iou[i, j]:.2f}", ha="center", va="center",
                            fontsize=5.5, color="w" if iou[i, j] < 0.6 else "k")
        ax.set_title(f"[{lab}] pairwise soft-IoU  mean={off.mean():.2f}", fontsize=8)
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.tick_params(labelsize=5)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle("Do the K parts differentiate?  (std≈0 / IoU≈1 ⇒ all parts are copies)", fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(a.out, dpi=140, bbox_inches="tight")
    print(f"已保存 {a.out}")


if __name__ == "__main__":
    main()
