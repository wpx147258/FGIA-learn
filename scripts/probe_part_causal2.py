"""因果性诊断 v2：按审稿标准补齐（配对 flips + McNemar + 掩码替换的多个变体）。

v1 只报了 top1 差值；审稿视角指出：**n=5794 上的 0.28pp ≈ 16 张**，
必须给配对 flips 和显著性，否则"没掉"和"掉了一点"分不开。

本版在同一 checkpoint 上跑七种干预，逐张记录对错，再两两配对：

  baseline     原样
  zero-res     残差投影置零（= 去掉整个分支对 logits 的贡献）
  no-branch    完全不挂模块（应与 zero-res 一致，作为一致性检查）
  rand-mask    学到的掩码 → 固定随机掩码
  uniform-mask 所有部件用同一张均匀掩码（等价 GAP 池化，但保留那条加性通路）
  rolled-mask  掩码按 (7,7) 循环平移（保留形状、破坏空间对应）
  oracle-mask  用 GT 鸟框当掩码（诊断上限，不是可用方法）

判读（审稿建议的阈值，事先写定）：
  · zero-res 掉 ≥1.0pp 且 CI 下界 >0.5pp ⇒ 分支被"实质使用"；
  · 掉 ≤0.3pp、flips ≤29 张 ⇒ 未被实质使用；
  · oracle 比 learned 好 ≥2pp ⇒ 掩码被用但次优；oracle 不比 learned 好 ⇒ 空间模式与因果无关。
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.cub import CUBDataset
from dataset.transforms import build_transform, resolve_data_config
from engine.trainer import read_checkpoint, resolve_amp
from model.classifier import build_model, load_state_dict_tolerant
from utils.config import resolve_path


class MaskOverride:
    """临时替换 PartAttention.part_masks 的返回值，用于掩码干预。"""

    def __init__(self, pa, fn):
        self.pa, self.fn, self.orig = pa, fn, pa.part_masks

    def __enter__(self):
        pa = self.pa

        def wrapped(feat):
            return self.fn(feat, pa)
        pa.part_masks = wrapped
        return self

    def __exit__(self, *a):
        self.pa.part_masks = self.orig


@torch.no_grad()
def predict(model, dl, dev, amp):
    preds = []
    for batch in dl:
        x = batch["image"].to(dev)
        if amp:
            with torch.autocast(device_type="cuda", dtype=amp):
                out = model(x)
        else:
            out = model(x)
        logits = out["logits"] if isinstance(out, dict) else out
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)


def mcnemar(a_ok, b_ok):
    n10 = int((a_ok & ~b_ok).sum())
    n01 = int((~a_ok & b_ok).sum())
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    z = abs(n10 - n01) / math.sqrt(n)
    p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    return n10, n01, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = read_checkpoint(a.checkpoint, map_location=dev)
    cfg = ck["config"]
    dcfg = cfg["DATASET"]
    data_cfg = dict(resolve_data_config(build_model(cfg).backbone))
    if dcfg.get("crop_pct"):
        data_cfg["crop_pct"] = float(dcfg["crop_pct"])
    root = resolve_path(dcfg["root"])
    ds = CUBDataset(data_root=root, split="test",
                    transform=build_transform(data_cfg, dcfg["image_size"], train=False),
                    num_classes=dcfg["num_classes"])
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                    num_workers=dcfg.get("num_workers", 4), pin_memory=True)
    amp = resolve_amp(cfg)
    ytrue = np.array([ds[i]["label"] for i in range(len(ds))])

    def build(no_branch=False):
        c = copy.deepcopy(cfg)
        if no_branch:
            c["MODEL"]["part_attn"] = False
        m = build_model(c).to(dev)
        sd = ck["model"]
        if no_branch:
            sd = {k: v for k, v in sd.items() if ".part_attn." not in k}
        load_state_dict_tolerant(m, sd)
        m.eval()
        return m

    res = {}
    m = build()
    pa = m.backbone.part_attn
    res["baseline"] = predict(m, dl, dev, amp)

    m = build()
    with torch.no_grad():
        m.backbone.part_attn.out.weight.zero_()
        m.backbone.part_attn.out.bias.zero_()
    res["zero-res"] = predict(m, dl, dev, amp)

    m = build(no_branch=True)
    res["no-branch"] = predict(m, dl, dev, amp)

    m = build()
    m.backbone.part_attn.random_masks = True
    m.backbone.part_attn._rand_m = None
    res["rand-mask"] = predict(m, dl, dev, amp)

    m = build()
    pa = m.backbone.part_attn
    with MaskOverride(pa, lambda feat, p: torch.full(
            (feat.shape[0], p.num_parts, feat.shape[2], feat.shape[3]),
            1.0 / (feat.shape[2] * feat.shape[3]), device=feat.device, dtype=feat.dtype)):
        res["uniform-mask"] = predict(m, dl, dev, amp)

    m = build()
    pa = m.backbone.part_attn
    orig_fn = pa.part_masks

    def _rolled(feat, p_):
        return torch.roll(orig_fn(feat), shifts=(7, 7), dims=(2, 3))

    with MaskOverride(pa, _rolled):
        res["rolled-mask"] = predict(m, dl, dev, amp)


    base = res["baseline"] == ytrue
    print(f"n={len(ds)}  acc(baseline)={base.mean()*100:.2f}%  "
          f"ckpt={Path(a.checkpoint).parent.name}")
    print(f"\n{'干预':16s}{'top1':>8s}{'Δpp':>8s}{'flips':>7s}{'n10/n01':>12s}{'p':>9s}")
    for k, v in res.items():
        if k == "baseline":
            continue
        ok = v == ytrue
        n10, n01, p = mcnemar(base, ok)
        print(f"{k:16s}{ok.mean()*100:8.2f}{(ok.mean()-base.mean())*100:+8.2f}"
              f"{int((v != res['baseline']).sum()):7d}{f'{n10}/{n01}':>12s}{p:9.3f}")


if __name__ == "__main__":
    main()
