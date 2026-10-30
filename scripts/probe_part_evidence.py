"""零训练诊断：部件分支的逐部件 logits 到底带不带「独立证据」。

背景（2026-10-30）：
- 错误诊断显示 8% 的 test 样本真类已在 top-5 但没排第一，margin 中位仅 −1.1；
- 熵上限实验证明掩码可以训练成真部件（熵 3.30、重叠 0.009），但准确率反降；
- 于是剩下的唯一问题是：**部件证据本身有没有用**，还是它只是主干的复印件。

做法：对 test 集跑一次前向，同时取出
  主 logits                      —— 分类头
  逐部件 logits（part_cls）      —— 每个部件单独分类的结果  [B,K,200]
然后回答三件事：
  1. 把 K 个部件 logits 平均（或取最大）单独分类，top1 是多少？—— 部件证据的绝对质量
  2. 在「主分支错但在 top-5」的那些样本上，部件证据的 top1 猜对多少？—— 它能不能救排名错误
  3. 部件证据与主分支的预测有多一致？—— 它是不是复印件

用法：
    python scripts/probe_part_evidence.py --checkpoint outputs/<run>/<ts>/swa_f0.5.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.cub import CUBDataset
from dataset.transforms import build_transform, resolve_data_config
from engine.trainer import read_checkpoint, resolve_amp
from model.classifier import build_model, load_state_dict_tolerant
from utils.config import resolve_path
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = read_checkpoint(a.checkpoint, map_location=dev)
    cfg = ck["config"]
    model = build_model(cfg).to(dev)
    load_state_dict_tolerant(model, ck["model"])
    model.eval()

    dcfg = cfg["DATASET"]
    data_cfg = dict(resolve_data_config(model.backbone))
    if dcfg.get("crop_pct"):
        data_cfg["crop_pct"] = float(dcfg["crop_pct"])
    ds = CUBDataset(data_root=resolve_path(dcfg["root"]), split=a.split,
                    transform=build_transform(data_cfg, dcfg["image_size"], train=False),
                    num_classes=dcfg["num_classes"])
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    pa = getattr(model.backbone, "part_attn", None)
    if pa is None or getattr(pa, "part_cls", None) is None:
        sys.exit("该 checkpoint 没有部件分类头（part_cls），无法做这个诊断")

    amp_dtype = resolve_amp(cfg)
    mains, parts, ys = [], [], []
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(dev, non_blocking=True)
            y = batch["label"].to(dev)
            with torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype else torch.no_grad():
                lg = model(x)
            if isinstance(lg, dict):
                lg = lg["logits"]
            mains.append(lg.float().cpu())
            parts.append(pa._last_part_logits.float().cpu())   # [B,K,200]
            ys.append(y.cpu())

    main_lg = torch.cat(mains)
    part_lg = torch.cat(parts)          # [N,K,C]
    y = torch.cat(ys)
    n = len(y)

    main_top1 = main_lg.argmax(1)
    acc_main = (main_top1 == y).float().mean().item()
    top5 = main_lg.topk(5, dim=1).indices
    in5 = (top5 == y[:, None]).any(1)

    part_mean = part_lg.mean(1)
    part_max = part_lg.max(1).values
    acc_pm = (part_mean.argmax(1) == y).float().mean().item()
    acc_px = (part_max.argmax(1) == y).float().mean().item()

    print(f"n={n}")
    print(f"主分支            top1 = {acc_main*100:.2f}%")
    print(f"部件 logits 平均   top1 = {acc_pm*100:.2f}%")
    print(f"部件 logits 取最大 top1 = {acc_px*100:.2f}%")

    # 部件证据的独立性：与主分支 top1 的一致率
    agree = (part_mean.argmax(1) == main_top1).float().mean().item()
    print(f"部件平均与主分支预测一致率 = {agree*100:.2f}%")

    # 关键：主分支错但真类在 top5 的样本上，部件证据能不能救
    hard = in5 & (main_top1 != y)
    print(f"\n主分支错但在 top-5 的样本: {int(hard.sum())} 张 ({hard.float().mean().item()*100:.2f}%)")
    if hard.sum():
        rescued = (part_mean[hard].argmax(1) == y[hard]).float().mean().item()
        rescued5 = (part_mean[hard].topk(5, dim=1).indices == y[hard, None]).any(1).float().mean().item()
        print(f"  部件平均在这些样本上 top1 就猜对: {rescued*100:.2f}%")
        print(f"  部件平均在这些样本上 top5 里有真类: {rescued5*100:.2f}%")
        # 参照：这些样本上「随机选 top5 里的一个」
        print(f"  （主分支自己 top5 覆盖 = 100%，随机重排只有约 20%）")


if __name__ == "__main__":
    main()
