"""零训练诊断：按 top-1 / top-2 预测类各算一张 CAM，分别裁出来重编码，量 oracle 上限。

动机：cam_fuse（只裁 top-1 的 CAM 框）实测 −0.59，说明「最判别区域」和整图高度相关，
提供不了互补证据。但不同类别假设的 CAM 指向不同的证据位置——如果 top-2 的框能救回
一批整图错的样本，那么「多假设区域重编码 + 学习式融合」就有空间；否则这条路到头了。

用法：
  python scripts/probe_zoom2.py --checkpoint <ckpt> [--batch-size 24]
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
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--margin", type=float, default=1.3)
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    dev = torch.device("cuda")
    ck = read_checkpoint(args.checkpoint, map_location=dev)
    cfg = ck["config"]; amp = resolve_amp(cfg)
    model = build_model(cfg).to(dev); model.load_state_dict(ck["model"]); model.eval()

    dcfg = cfg["DATASET"]
    data_cfg = dict(resolve_data_config(model.backbone))
    if dcfg.get("crop_pct"):
        data_cfg["crop_pct"] = float(dcfg["crop_pct"])
    ds = CUBDataset(resolve_path(dcfg["root"]), args.split,
                    build_transform(data_cfg, dcfg["image_size"], train=False),
                    dcfg["num_classes"])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=dcfg["num_workers"], pin_memory=True)
    size = int(dcfg["image_size"])

    P = {k: [] for k in ("global", "c1", "c2")}
    labels = []
    for batch in loader:
        x = batch["image"].to(dev); labels.append(batch["label"])
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            feat = model.backbone.forward_features(x)
            logits = model(x)
        probs = logits.float().softmax(1)
        top2 = probs.topk(2, dim=1).indices
        w = model.head.cam_weight(feat.shape[1]).detach()
        for key, cls in (("c1", top2[:, 0]), ("c2", top2[:, 1])):
            cam = cam_map(feat.float(), w, cls)
            boxes = mask_to_box(peak_mask(cam, args.alpha).detach())
            patch = crop_and_resize(x, boxes, size, args.margin)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                lg = model(patch)
            P[key].append(lg.float().softmax(1).cpu())
        P["global"].append(probs.cpu())

    y = torch.cat(labels)
    Q = {k: torch.cat(v) for k, v in P.items()}
    acc = {k: float((v.argmax(1) == y).float().mean() * 100) for k, v in Q.items()}
    ok = {k: Q[k].argmax(1) == y for k in Q}
    for k in ("global", "c1", "c2"):
        print(f"{k:6s} top1 = {acc[k]:.2f}")
    orc = ok["global"] | ok["c1"] | ok["c2"]
    print(f"oracle top1 = {float(orc.float().mean() * 100):.2f}  （global + top1/top2 两处裁剪）")
    print(f"  global 错、c1/c2 至少一个对: {int((~ok['global'] & (ok['c1'] | ok['c2'])).sum())}")
    print(f"  c1 对、global 错: {int((~ok['global'] & ok['c1']).sum())}   "
          f"c2 对、global 错: {int((~ok['global'] & ok['c2']).sum())}")


if __name__ == "__main__":
    main()
