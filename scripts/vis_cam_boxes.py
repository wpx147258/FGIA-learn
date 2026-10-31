"""可视化 CAM 裁剪框——当前最优模型（e75m2 家族）真正在用的那个机制。

为什么画这个而不是部件框：项目里准确率最好的那批模型（e75m2 + drop_path / crop_scale）
**没有 part_attn**，它们的机制是 CAM 裁剪分支——开关值 +1.19，是全项目最大的单项。
所以对它们做"框可视化"，要画的是 CAM 框，而不是部件掩码。

框的定义与训练时**完全一致**（`model/cam_aug.py` 同一条路径）：
  1. 用**预测类**（不是 GT）的分类头权重对 layer4 特征做 CAM；
  2. `peak_mask(alpha)` 取 CAM 峰值的 alpha 以上；alpha 取配置里的 `cam_alpha_crop`；
  3. `mask_to_box` 取外接框，再按 `cam_margin` 外扩 —— 这就是训练时裁给第二路的那块。

画在同一坐标系（448 网络输入）上，并叠 GT 鸟框做参照，输出格式与
`vis_part_boxes.py` 一致（每格一张图 + 面积百分比）。

用法：
    python scripts/vis_cam_boxes.py --checkpoint A.pth [B.pth ...] --n 12 --out figs/xxx.png
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
from model.cam_aug import cam_map, mask_to_box, peak_mask
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


@torch.no_grad()
def boxes_for(ckpt_path, idxs, dev):
    ck = read_checkpoint(ckpt_path, map_location=dev)
    cfg = ck["config"]
    model = build_model(cfg).to(dev)
    load_state_dict_tolerant(model, ck["model"])
    model.eval()
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
    x = batch["image"].to(dev)
    amp = resolve_amp(cfg)
    feat = model.backbone.forward_features(x).float()
    logits = model.head(model.backbone.forward_head(feat))
    pred = logits.argmax(1)
    # 用 head 自己的 cam_weight(feat_dim) ——它处理了 BN 头、2C 拼接头等情形
    if not hasattr(model.head, "cam_weight"):
        sys.exit("该 head 不支持 cam_weight，无法做 CAM")
    w = model.head.cam_weight(feat.shape[1])
    cam = cam_map(feat, w.float(), pred)
    alpha = float(cfg["TRAIN"].get("cam_alpha_crop", 0.2))
    margin = float(cfg["TRAIN"].get("cam_margin", 1.3))
    m = peak_mask(cam, alpha)
    b = mask_to_box(m).cpu().numpy()
    mean = np.array(data_cfg["mean"], np.float32).reshape(3, 1, 1)
    std = np.array(data_cfg["std"], np.float32).reshape(3, 1, 1)
    imgs = (batch["image"].numpy() * std + mean).clip(0, 1).transpose(0, 2, 3, 1)
    correct = (pred == batch["label"].to(dev)).cpu().numpy()
    return dict(box=b, imgs=imgs, ds=ds, root=root, img_size=img_size,
                crop_pct=float(data_cfg.get("crop_pct") or 1.0),
                alpha=alpha, margin=margin, correct=correct, name=Path(ckpt_path).parent.parent.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="figs/cam_boxes.png")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idxs = np.linspace(0, 5793, a.n).astype(int).tolist()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from matplotlib.patches import Rectangle
    from matplotlib import patheffects as pe

    runs = [boxes_for(p, idxs, dev) for p in a.checkpoint]
    R = len(runs)
    fig, axes = plt.subplots(R, a.n, figsize=(2.0 * a.n, 2.3 * R))
    axes = np.atleast_2d(axes)
    halo = [pe.Stroke(linewidth=2.4, foreground="white"), pe.Normal()]

    print(f"alpha(cam_alpha_crop)={runs[0]['alpha']}  margin(cam_margin)={runs[0]['margin']}")
    for r, run in enumerate(runs):
        ds, root, img_size, crop_pct = run["ds"], run["root"], run["img_size"], run["crop_pct"]
        gts, paths = load_gt(root)
        areas, gts_inside = [], []
        for bi in range(a.n):
            iid = ds.data_ids[idxs[bi]]
            W, H = Image.open(root / "images" / paths[iid]).size
            s = int(round(img_size / crop_pct)) / min(W, H)
            nw, nh = ((int(round(int(round(img_size / crop_pct)) * W / H)), int(round(img_size / crop_pct)))
                      if W >= H else (int(round(img_size / crop_pct)),
                                      int(round(int(round(img_size / crop_pct)) * H / W))))
            ox, oy = (nw - img_size) / 2.0, (nh - img_size) / 2.0
            gx, gy, gw, gh = gts[iid]
            gx0, gy0 = gx * s - ox, gy * s - oy
            gbw, gbh = gw * s, gh * s

            x0, y0, x1, y1 = run["box"][bi]
            # mask_to_box 给的是归一化坐标；按 margin 外扩（与 crop_and_resize 一致：中心 + 边长×margin）
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            hw, hh = (x1 - x0) * run["margin"] / 2, (y1 - y0) * run["margin"] / 2
            nx0, ny0 = max(cx - hw, 0.0) * img_size, max(cy - hh, 0.0) * img_size
            nx1, ny1 = min(cx + hw, 1.0) * img_size, min(cy + hh, 1.0) * img_size
            area = (nx1 - nx0) * (ny1 - ny0) / (img_size * img_size)
            areas.append(area)
            inter = max(0.0, min(nx1, gx0 + gbw) - max(nx0, gx0)) * max(0.0, min(ny1, gy0 + gbh) - max(ny0, gy0))
            gts_inside.append(inter / max((nx1 - nx0) * (ny1 - ny0), 1e-9))

            ax = axes[r][bi]
            ax.imshow(run["imgs"][bi])
            ax.add_patch(Rectangle((gx0, gy0), gbw, gbh, fill=False, ec="lime",
                                   lw=1.3, ls="--", path_effects=halo, zorder=5))
            ax.add_patch(Rectangle((nx0, ny0), nx1 - nx0, ny1 - ny0, fill=False,
                                   ec="#00E5FF", lw=1.7, path_effects=halo, zorder=6))
            ax.set_xticks([]); ax.set_yticks([])
            if bi == 0:
                ax.set_ylabel(run["name"][:26], fontsize=6)
            ax.set_title(f"{area*100:.0f}% | GT {gts_inside[-1]*100:.0f}%", fontsize=5.5)
        print(f"[{run['name']}] 裁剪框面积中位 {np.median(areas)*100:.1f}%  "
              f"框内 GT 覆盖率中位 {np.median(gts_inside)*100:.1f}%  "
              f"该格 top1={run['correct'].mean()*100:.1f}%")

    fig.suptitle("CAM crop boxes (cyan) vs GT bird box (green) — what the best models actually crop",
                 fontsize=8)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"已保存 {a.out}")


if __name__ == "__main__":
    main()
