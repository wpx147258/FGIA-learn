"""因果性诊断：部件分支到底有没有被分类器用上（推理期干预，不训练）。

前情：三种部件掩码配置的几何差异巨大（框 29%~100%、K_eff 1.95~7.0、GT 富集 0.98~1.15），
但分类准确率几乎一样（89.1~89.8）。这只说明"几何不影响精度"，
**没有说明分支是否被使用**——也可能它一直在起作用，只是这些几何量无关。

三个推理期干预，逐个回答一个问题：

  baseline  原样前向
  zero-res  把残差投影 out 置零 ⇒ 部件分支对 logits 的贡献恒为 0
            （权重不同、其余不变；若 acc 不变 ⇒ 分支对决策无因果贡献）
  rand-mask 把学到的掩码换成固定随机掩码（模块自带 pa_random_masks）
            （若 acc 不变 ⇒ 学习到的掩码内容无关，只剩"多一条通路"这件事本身）
  no-branch 完全不挂 part_attn（真正的下界：去掉整个模块）

判读：
  · zero-res 掉明显 ⇒ 分支是**因果使用**的，值得继续改进它提供的证据；
  · zero-res 不掉、rand-mask 也不掉 ⇒ 分支只是"多一条加性通路"（类无关偏置），
    掩码内容无关，这条路在方法学上已到头。

用法：
    python scripts/probe_part_causal.py --checkpoint <ckpt>
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.cub import CUBDataset
from dataset.transforms import build_transform, resolve_data_config
from engine.trainer import read_checkpoint, resolve_amp
from model.classifier import build_model, load_state_dict_tolerant
from utils.config import resolve_path


@torch.no_grad()
def accuracy(model, dl, dev, amp):
    ok = n = 0
    for batch in dl:
        x = batch["image"].to(dev)
        if amp:
            with torch.autocast(device_type="cuda", dtype=amp):
                out = model(x)
        else:
            out = model(x)
        logits = out["logits"] if isinstance(out, dict) else out
        ok += (logits.argmax(1).cpu() == batch["label"]).sum().item()
        n += x.shape[0]
    return ok / n * 100, ok, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = read_checkpoint(a.checkpoint, map_location=dev)
    cfg = ck["config"]
    dcfg = cfg["DATASET"]
    data_cfg = dict(resolve_data_config(build_model(cfg).backbone))
    if dcfg.get("crop_pct"):
        data_cfg["crop_pct"] = float(dcfg["crop_pct"])
    ds = CUBDataset(data_root=resolve_path(dcfg["root"]), split=a.split,
                    transform=build_transform(data_cfg, dcfg["image_size"], train=False),
                    num_classes=dcfg["num_classes"])
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                    num_workers=dcfg.get("num_workers", 4), pin_memory=True)
    amp = resolve_amp(cfg)
    print(f"n={len(ds)}  amp={amp}  ckpt={Path(a.checkpoint).parent.name}")

    def build():
        m = build_model(copy.deepcopy(cfg)).to(dev)
        load_state_dict_tolerant(m, ck["model"])
        m.eval()
        return m

    res = {}

    m = build()
    pa = getattr(m.backbone, "part_attn", None)
    if pa is None:
        sys.exit("该 checkpoint 没有 part_attn")
    res["baseline"] = accuracy(m, dl, dev, amp)[0]

    # ① 残差投影置零
    m = build()
    pa = m.backbone.part_attn
    with torch.no_grad():
        pa.out.weight.zero_(); pa.out.bias.zero_()
    res["zero-res"] = accuracy(m, dl, dev, amp)[0]

    # ② 学到的掩码 → 固定随机掩码
    m = build()
    pa = m.backbone.part_attn
    pa.random_masks = True
    pa._rand_m = None
    res["rand-mask"] = accuracy(m, dl, dev, amp)[0]

    # ③ 完全不挂分支（下界）
    cfg2 = copy.deepcopy(cfg)
    cfg2["MODEL"]["part_attn"] = False
    m = build_model(cfg2).to(dev)
    # 不挂分支时，checkpoint 里的 part_attn.* 是"多余键"，先剔掉再严格加载
    sd = {k: v for k, v in ck["model"].items() if ".part_attn." not in k}
    load_state_dict_tolerant(m, sd)
    m.eval()
    res["no-branch"] = accuracy(m, dl, dev, amp)[0]

    base = res["baseline"]
    print(f"\n{'配置':14s}{'top1':>8s}{'Δ vs baseline':>16s}")
    for k, v in res.items():
        print(f"{k:14s}{v:8.2f}{v - base:+16.2f}")


if __name__ == "__main__":
    main()
