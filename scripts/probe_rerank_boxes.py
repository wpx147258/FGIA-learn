"""诊断：部件重编码（part_rerank）在测试时到底裁了多大、多不重叠的区域。

如果两个部件的框都接近整图（面积 ~1）或彼此 IoU 很高，那么"重编码"实际上
只是把整图又看了一遍，机制没有产生任何放大；这能解释为什么会掉点。

用法：
  python scripts/probe_rerank_boxes.py --checkpoint outputs/r50_rerank3/<ts>/swa_f0.5.pth
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
from model.cam_aug import cam_map, mask_to_box, peak_mask
from model.classifier import build_model
from utils.config import resolve_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--alpha", type=float, default=0.2)
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
    pa = model.backbone.part_attn
    areas, ious, cam_ious = [], [], []
    for batch in loader:
        x = batch["image"].to(dev)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            feat = model.backbone.forward_features(x)
            masks = pa.part_masks(feat)
            logits = model.head(model.backbone.forward_head(feat))
        masks = masks.float(); feat = feat.float()
        b, k = masks.shape[0], masks.shape[1]
        if 0 < model.pr_parts < k:
            mass = masks.flatten(2).sum(-1)
            idx = mass.topk(model.pr_parts, dim=1).indices
            masks = torch.gather(masks, 1, idx[..., None, None].expand(-1, -1, *masks.shape[2:]))
            k = model.pr_parts
        boxes = mask_to_box(peak_mask(masks.reshape(b * k, *masks.shape[2:]), args.alpha).detach()).view(b, k, 4)
        wh = (boxes[..., 2:] - boxes[..., :2]).clamp_min(0)
        areas.append((wh[..., 0] * wh[..., 1]).flatten().cpu())
        if k > 1:
            a = wh[..., 0] * wh[..., 1]
            ix = (torch.minimum(boxes[:, :1, 2], boxes[:, 1:, 2]) -
                  torch.maximum(boxes[:, :1, 0], boxes[:, 1:, 0])).clamp_min(0)
            iy = (torch.minimum(boxes[:, :1, 3], boxes[:, 1:, 3]) -
                  torch.maximum(boxes[:, :1, 1], boxes[:, 1:, 1])).clamp_min(0)
            inter = ix * iy
            iou = inter / (a[:, :1] + a[:, 1:] - inter).clamp_min(1e-6)
            ious.append(iou.flatten().cpu())
        # 与预测类 CAM 框的重合
        fm = model._to_feature_map(feat)
        pred = logits.float().argmax(1)
        w = model.head.cam_weight(fm.shape[1]).detach()
        cam = cam_map(fm.float(), w, pred)
        cb = mask_to_box(peak_mask(cam, 0.2).detach())
        cw = (cb[:, 2:] - cb[:, :2]).clamp_min(0)
        ca = cw[:, 0] * cw[:, 1]
        for j in range(k):
            bb = boxes[:, j]
            ix = (torch.minimum(bb[:, 2], cb[:, 2]) - torch.maximum(bb[:, 0], cb[:, 0])).clamp_min(0)
            iy = (torch.minimum(bb[:, 3], cb[:, 3]) - torch.maximum(bb[:, 1], cb[:, 1])).clamp_min(0)
            inter = ix * iy
            cam_ious.append((inter / (ca + (bb[:, 2]-bb[:, 0])*(bb[:, 3]-bb[:, 1]) - inter)
                             .clamp_min(1e-6)).flatten().cpu())
    A = torch.cat(areas)
    print(f"部件框面积占比: mean={float(A.mean()):.3f}  median={float(A.median()):.3f}  "
          f">0.6 的比例={float((A>0.6).float().mean()):.3f}")
    if ious:
        I = torch.cat(ious)
        print(f"两部件框 IoU: mean={float(I.mean()):.3f}  >0.5 的比例={float((I>0.5).float().mean()):.3f}")
    C = torch.cat(cam_ious)
    print(f"部件框与 CAM 框 IoU: mean={float(C.mean()):.3f}")


if __name__ == "__main__":
    main()
