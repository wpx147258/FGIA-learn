"""零训练诊断：测试时如果多一路「按 CAM 框放大后再看一次」，还能救回多少？

背景
----
项目里唯一有效的非主干机制是 CAM 裁剪分支 + 一致性（训练期把判别区放大后
再看一次，+0.79/+0.8）。但测试时只有整图一次前向 —— 也就是说，训练时见过的
「放大视角」在测试时没有被用上。

这个脚本不解这个问题，只量化它的上限：
  1. global  ：整图（当前论文口径）
  2. crop    ：按模型自己的 CAM 框 + margin 裁出来，缩到 448 再看一次
  3. avg     ：两路概率平均（相当于一个两视图的小 TTA）
  4. oracle  ：两路里只要有一路答对就算对（任何融合方式的理论上限）
  5. 若 crop 单独看很强、且 oracle 明显高于 global，说明「测试时放大」值得做成
     结构（学习式区域提议 / STN / 单前向内的 zoom 分支）；否则这条路直接排除。

注意：crop 用的是 **预测类** 的 CAM（测试时没有标签），和训练时的口径一致。
这不是 TTA 结论，只是诊断；不写进最终成绩。

用法：
    python scripts/probe_zoom.py --checkpoint outputs/<run>/<ts>/swa_f0.5.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.cub import CUBDataset
from dataset.transforms import build_transform, resolve_data_config
from engine.trainer import read_checkpoint, resolve_amp
from model.cam_aug import cam_map, crop_and_resize, mask_to_box, peak_mask
from model.classifier import build_model
from utils.config import resolve_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--alpha", type=float, default=0.2, help="峰值阈值（与训练一致）")
    p.add_argument("--margin", type=float, default=1.3, help="框外扩（与训练一致）")
    p.add_argument("--dump", default="", help="概率前缀，存 <prefix>_global.npy / _crop.npy")
    p.add_argument("--batch-size", type=int, default=0,
                   help="覆盖 EVAL.batch_size（显存紧时调小）")
    return p.parse_args()


def make_loader(cfg, model, split):
    dcfg = cfg["DATASET"]
    data_cfg = dict(resolve_data_config(model.backbone))
    ds = CUBDataset(
        data_root=resolve_path(dcfg["root"]),
        split=split,
        transform=build_transform(data_cfg, dcfg["image_size"], train=False),
        num_classes=dcfg["num_classes"],
    )
    return DataLoader(ds, batch_size=cfg["EVAL"]["batch_size"], shuffle=False,
                      num_workers=dcfg["num_workers"], pin_memory=True)


@torch.no_grad()
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = read_checkpoint(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    amp_dtype = resolve_amp(cfg)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if args.batch_size:
        cfg = {**cfg, "EVAL": {**cfg["EVAL"], "batch_size": args.batch_size}}
    loader = make_loader(cfg, model, args.split)
    size = int(cfg["DATASET"]["image_size"])
    glob_p, crop_p, labels = [], [], []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"]
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            feat = model.backbone.forward_features(x)
            logits = model(x)
        # 预测类的 CAM（测试时没有标签）
        pred = logits.float().argmax(dim=1)
        w = model.head.cam_weight(feat.shape[1]).detach()
        cam = cam_map(feat.float(), w, pred)
        boxes = mask_to_box(peak_mask(cam, args.alpha).detach())
        patch = crop_and_resize(x, boxes, size, args.margin)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits_c = model(patch)
        glob_p.append(logits.float().softmax(1).cpu())
        crop_p.append(logits_c.float().softmax(1).cpu())
        labels.append(y)

    g = torch.cat(glob_p); c = torch.cat(crop_p); y = torch.cat(labels)

    def acc(p):
        return (p.argmax(1) == y).float().mean().item() * 100

    gok = g.argmax(1) == y
    cok = c.argmax(1) == y
    print(f"global top1 = {acc(g):.2f}")
    print(f"crop   top1 = {acc(c):.2f}")
    print(f"avg    top1 = {acc((g + c) / 2):.2f}   （两视图平均，仅作参照）")
    print(f"oracle top1 = {(gok | cok).float().mean().item() * 100:.2f}   "
          f"（两路取并集的上限）")
    print(f"  global 对 crop 错: {(gok & ~cok).sum().item()}")
    print(f"  global 错 crop 对: {(~gok & cok).sum().item()}")
    print(f"  两路都错: {(~gok & ~cok).sum().item()}")

    if args.dump:
        import numpy as np
        np.save(f"{args.dump}_global.npy", g.numpy())
        np.save(f"{args.dump}_crop.npy", c.numpy())
        np.save(f"{args.dump}_labels.npy", y.numpy())
        print("dumped ->", args.dump)


if __name__ == "__main__":
    run(parse_args())
