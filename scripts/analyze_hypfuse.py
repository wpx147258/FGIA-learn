"""分析假设条件重检（hyp_fuse）到底做了什么。

对训练好的 checkpoint 在 test 上分别统计：
  global        整图单独
  hyp1 / hyp2   top-1 / top-2 类别的 CAM 框裁剪后单独
  fused         global + β1·hyp1 + β2·hyp2（模型的最终输出）
  翻转          fused 改判的样本里，改对多少、改错多少

用法：
  python scripts/analyze_hypfuse.py --checkpoint outputs/r50_hypfuse/<ts>/swa_f0.5.pth
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
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dump", default="", help="保存四路 logits 供离线门控实验")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    dev = torch.device("cuda")
    ck = read_checkpoint(args.checkpoint, map_location=dev)
    cfg = ck["config"]; amp = resolve_amp(cfg)
    model = build_model(cfg).to(dev); model.load_state_dict(ck["model"]); model.eval()
    b1 = float(model.hyp_beta1); b2 = float(model.hyp_beta2)
    print(f"learned betas: beta1={b1:+.4f}  beta2={b2:+.4f}")

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

    out = {k: [] for k in ("global", "hyp1", "hyp2", "fused")}
    ys = []
    for batch in loader:
        x = batch["image"].to(dev); ys.append(batch["label"])
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            feat = model.backbone.forward_features(x)
            lg = model.head(model.backbone.forward_head(feat))
        fm = model._to_feature_map(feat)
        top2 = lg.detach().float().topk(2, 1).indices
        w = model.head.cam_weight(fm.shape[1]).detach()
        hs = []
        for j in range(2):
            cam = cam_map(fm.float(), w, top2[:, j])
            boxes = mask_to_box(peak_mask(cam, model.hyp_alpha).detach())
            patch = crop_and_resize(x, boxes, model.hyp_size, model.hyp_margin)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                hs.append(model.head(model.backbone.forward_head(
                    model.backbone.forward_features(patch))))
        out["global"].append(lg.float().cpu())
        out["hyp1"].append(hs[0].float().cpu())
        out["hyp2"].append(hs[1].float().cpu())
        out["fused"].append((lg + b1 * hs[0] + b2 * hs[1]).float().cpu())

    y = torch.cat(ys)
    P = {k: torch.cat(v) for k, v in out.items()}
    if args.dump:
        import numpy as np
        for k, v in P.items():
            np.save(f"{args.dump}_{k}.npy", v.numpy())
        np.save(f"{args.dump}_labels.npy", y.numpy())
        print("dumped ->", args.dump)
    ok = {k: P[k].argmax(1) == y for k in P}
    for k in ("global", "hyp1", "hyp2", "fused"):
        print(f"{k:7s} top1 = {float(ok[k].float().mean()*100):.2f}")
    flipped = ok["fused"] != ok["global"]
    print(f"翻转样本 {int(flipped.sum())} 个：改对 {int((flipped & ok['fused']).sum())}、"
          f"改错 {int((flipped & ~ok['fused']).sum())}")
    print(f"oracle(global,hyp1,hyp2) = "
          f"{float((ok['global']|ok['hyp1']|ok['hyp2']).float().mean()*100):.2f}")


if __name__ == "__main__":
    main()
