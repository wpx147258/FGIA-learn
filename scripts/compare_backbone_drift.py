"""比较训练后的 backbone 相对 ImageNet 初始权重偏离了多少。

动机：不同 head 的收敛速度不同（NetVLAD 的 loss 在 ep7 还是 3.1，linear 已经 1.7），
     高损失 → 大梯度 → backbone 被推着走的**时间更长**。
     这个脚本用来回答：那个更长的"推"到底有没有真的伤到预训练权重。

用法：
    python scripts/compare_backbone_drift.py <ckpt> [<ckpt> ...]

每个 checkpoint 会和同一个 ImageNet 初始权重比，输出：
  * 全局相对 L2 偏离  ||w - w0|| / ||w0||
  * 全局余弦相似度
  * 按 stage（layer1..4 / 其它）分组的偏离
  * 偏离最大的 5 个参数

不加载到 GPU，也不需要数据。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import timm
from engine.trainer import read_checkpoint


def stage_of(name: str) -> str:
    for s in ("layer1", "layer2", "layer3", "layer4"):
        if f".{s}." in f".{name}.":
            return s
    return "stem/other"


@torch.no_grad()
def drift(ckpt_path: Path, init_sd: dict) -> dict:
    ck = read_checkpoint(ckpt_path, map_location="cpu")
    sd = ck["model"]
    rows = []
    for k, w0 in init_sd.items():
        key = f"backbone.{k}"
        if key not in sd:
            continue
        w = sd[key]
        if w.shape != w0.shape or not w.dtype.is_floating_point:
            continue
        n0 = w0.float().norm()
        if n0 < 1e-8:
            continue
        rows.append({
            "name": k, "stage": stage_of(k),
            "rel": float((w.float() - w0.float()).norm() / n0),
            "cos": float(torch.nn.functional.cosine_similarity(
                w.float().flatten(), w0.float().flatten(), dim=0)),
            "numel": w.numel(),
        })

    # 全局：把所有参数拼起来算一个总偏离
    tot = sum(r["numel"] for r in rows)
    glob_rel = sum(r["rel"] ** 2 * r["numel"] for r in rows) ** 0.5 / tot ** 0.5
    glob_cos = sum(r["cos"] * r["numel"] for r in rows) / tot

    by_stage: dict[str, list] = {}
    for r in rows:
        by_stage.setdefault(r["stage"], []).append(r)

    return {"path": ckpt_path, "epoch": ck.get("epoch"), "rows": rows,
            "glob_rel": glob_rel, "glob_cos": glob_cos, "by_stage": by_stage}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", type=Path)
    ap.add_argument("--backbone", default="resnet50.a1_in1k")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    print(f"加载初始权重 {args.backbone} ...")
    base = timm.create_model(args.backbone, pretrained=True, num_classes=0)
    init_sd = {k: v.detach().clone() for k, v in base.state_dict().items()}

    results = [drift(p, init_sd) for p in args.ckpts]

    print(f"\n{'checkpoint':<52} {'ep':>4} {'rel L2':>9} {'cos':>8}")
    print("-" * 78)
    for r in results:
        print(f"{str(r['path']):<52} {str(r['epoch']):>4} "
              f"{r['glob_rel']:9.4f} {r['glob_cos']:8.4f}")

    stages = ["stem/other", "layer1", "layer2", "layer3", "layer4"]
    print(f"\n{'按 stage 的相对偏离':<20} " +
          " ".join(f"{s:>11}" for s in stages))
    print("-" * 78)
    for r in results:
        cells = []
        for s in stages:
            g = r["by_stage"].get(s)
            if not g:
                cells.append(f"{'-':>11}")
            else:
                n = sum(x["numel"] for x in g)
                v = (sum(x["rel"] ** 2 * x["numel"] for x in g) / n) ** 0.5
                cells.append(f"{v:11.4f}")
        print(f"{Path(str(r['path'])).parent.name:<20} " + " ".join(cells))

    for r in results:
        top = sorted(r["rows"], key=lambda x: -x["rel"])[: args.top]
        print(f"\n{Path(str(r['path'])).parent.name} 偏离最大的 {args.top} 个参数：")
        for x in top:
            print(f"  {x['name']:<44} rel={x['rel']:.4f}  cos={x['cos']:.4f}")


if __name__ == "__main__":
    main()
