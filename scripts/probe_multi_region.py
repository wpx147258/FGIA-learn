"""零训练诊断：如果允许「一次调用、内部对若干区域再编码」
（即 2~3 次 backbone 前向），上限能到哪？

用 GT 的头部/身体框（只用于诊断上界，不能进最终方法）：
    global  —— 整图
    head    —— 头部框裁出、放大到 448 再前向
    body    —— 身体框裁出、放大到 448 再前向
    oracle  —— 三路里任一路答对就算对（任何融合方式的理论上限）
    avg     —— 三路概率平均（朴素融合，仅作参照）

判读：
  * oracle 明显高于 global（比如 +2 以上）→ 多区域再编码有真实信息增量，
    值得做学习式的区域提议 + 融合（2~3 次前向，属于机制要求，不是 TTA）；
  * oracle 只高 0.5 左右 → 这条路的上限就在那里，不值得投入。

注意：GT 框比任何学习式提议都好，所以这是**上界中的上界**。

用法：
  python scripts/probe_multi_region.py --checkpoint outputs/<run>/<ts>/swa_f0.5.pth
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
from model.classifier import build_model
from utils.config import resolve_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--batch-size", type=int, default=24)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = read_checkpoint(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    amp_dtype = resolve_amp(cfg)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dcfg = cfg["DATASET"]
    data_cfg = dict(resolve_data_config(model.backbone))
    if dcfg.get("crop_pct"):
        data_cfg["crop_pct"] = float(dcfg["crop_pct"])
    tf = build_transform(data_cfg, dcfg["image_size"], train=False)
    ds = CUBDataset(data_root=resolve_path(dcfg["root"]), split=args.split,
                    transform=tf, num_classes=dcfg["num_classes"],
                    use_parts=True, part_transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=dcfg["num_workers"], pin_memory=True)

    names = ["global", "head", "body"]
    probs = {n: [] for n in names}
    labels = []
    for batch in loader:
        labels.append(batch["label"])
        for n, key in zip(names, ["image", "head", "body"]):
            x = batch[key].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                lg = model(x)
            probs[n].append(lg.float().softmax(1).cpu())

    y = torch.cat(labels)
    P = {n: torch.cat(v) for n, v in probs.items()}

    def acc(p):
        return float((p.argmax(1) == y).float().mean() * 100)

    ok = {n: P[n].argmax(1) == y for n in names}
    for n in names:
        print(f"{n:6s} top1 = {acc(P[n]):.2f}")
    print(f"avg    top1 = {acc(sum(P.values()) / len(P)):.2f}   （朴素平均，仅参照）")
    orc = torch.zeros_like(ok["global"])
    for n in names:
        orc |= ok[n]
    print(f"oracle top1 = {float(orc.float().mean() * 100):.2f}   （三路并集上限）")
    print(f"  global 对、其它错: {int((ok['global'] & ~(ok['head'] | ok['body'])).sum())}")
    print(f"  global 错、head/body 对: {int((~ok['global'] & (ok['head'] | ok['body'])).sum())}")


if __name__ == "__main__":
    main()
