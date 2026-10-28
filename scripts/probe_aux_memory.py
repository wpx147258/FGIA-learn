"""测「训练期特权裁剪」多路前向的显存：主图 + head + 若干部位视图会不会 OOM。

只跑一步前向 + 反向，打印 torch 的峰值显存。用来在正式开跑前决定
「最多几路附加视图 / 是否需要 checkpoint」。

用法：python scripts/probe_aux_memory.py --views head,wing,tail --batch-size 16
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import load_config, BASE_CONFIG
from model.classifier import build_model
from engine.build import build_criterion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="head")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=448)
    a = ap.parse_args()
    dev = torch.device("cuda")
    ov = ["TRAIN.cam_crop=0.2", "TRAIN.cam_teacher=gt_head",
          "TRAIN.cam_crop_hard_label=true", f"DATASET.image_size={a.image_size}",
          f"TRAIN.cam_part_views={a.views}", "TRAIN.cam_part_w=0.3"]
    cfg = load_config([BASE_CONFIG, "configs/baseline_config.yaml"], ov)
    m = build_model(cfg).to(dev)
    crit = build_criterion(cfg)
    m.train()
    b = a.batch_size
    x = torch.randn(b, 3, a.image_size, a.image_size, device=dev)
    y = torch.randint(0, 200, (b,), device=dev)
    batch = {"label": y,
             "soft_target": torch.zeros(b, 200, device=dev).scatter_(1, y[:, None], 1.0)}
    for v in a.views.split(","):
        batch[v] = torch.randn(b, 3, a.image_size, a.image_size, device=dev)
    torch.cuda.reset_peak_memory_stats()
    o = m(x, y, batch)
    loss, _ = crit(o, batch)
    loss.backward()
    print(f"views={a.views} bs={b} size={a.image_size}  "
          f"峰值显存 {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, "
          f"保留 {torch.cuda.memory_reserved()/2**30:.2f} GiB")


if __name__ == "__main__":
    main()
