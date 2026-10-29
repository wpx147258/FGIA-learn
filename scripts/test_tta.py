"""测试时增强（TTA）：用同一个 checkpoint，把多路视图的概率平均后再判。

为什么先做这个：它不改训练、不占训练资源、不违反任何约束，
只是把已有权重测得更充分。报告里"多视图 +0.3~0.8"一直是估计，没有实测过。

四个视图（可以按 --views 选）：
    base   训练时用的评测流程：短边缩放到 448/crop_pct，中心裁 448
    flip   base 的水平翻转
    crop   换一个裁剪比例：1.0 = 短边直接缩到 448 不裁（看得更多）
           zoom = 0.75，裁得更紧（放大更多）
概率平均（不是 logit 平均）：不同视图的 logit 尺度不一致，softmax 后才可比。

用法：
    python scripts/test_tta.py --checkpoint outputs/<run>/<ts>/swa_f0.5.pth
    python scripts/test_tta.py --checkpoint ... --views base,flip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# 从仓库根以外的 cwd 跑时也能 import（与 probe_pretrained.py 一致）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.cub import CUBDataset
from dataset.transforms import build_transform, resolve_data_config
from engine.trainer import read_checkpoint, resolve_amp
from model.classifier import build_model, load_state_dict_tolerant
from utils.config import resolve_path


def parse_args():
    p = argparse.ArgumentParser(description="CUB 测试时增强")
    p.add_argument("--checkpoint", required=True, nargs="+",
                   help="一个或多个 checkpoint；多个时把各自的概率一起平均")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--views", default="base,flip,crop,zoom",
                   help="逗号分隔，可选 base / flip / crop / zoom")
    p.add_argument("--crop-pct", type=float, default=1.0,
                   help="视图 crop 的裁剪比例（1.0 = 不裁）")
    p.add_argument("--zoom-pct", type=float, default=0.75,
                   help="视图 zoom 的裁剪比例")
    p.add_argument("--dump", default="",
                   help="把融合后的逐样本概率存成 <前缀>_probs.npy / _labels.npy，"
                        "用于两个配置之间的配对检验（见 scripts/compare_runs.py）")
    return p.parse_args()


def make_loader(cfg, model, crop_pct: float | None, split: str) -> DataLoader:
    """按指定的 crop_pct 建一个评测 loader（None = 沿用 checkpoint 里的默认值）。"""
    dcfg = cfg["DATASET"]
    root = resolve_path(dcfg["root"])

    data_cfg = dict(resolve_data_config(model.backbone))
    if crop_pct is not None:
        data_cfg["crop_pct"] = crop_pct
    elif dcfg.get("crop_pct"):
        # 训练时如果显式统一过裁剪比例，评测必须沿用，否则两边视野不一致
        data_cfg["crop_pct"] = float(dcfg["crop_pct"])

    ds = CUBDataset(
        data_root=root,
        split=split,
        transform=build_transform(data_cfg, dcfg["image_size"], train=False),
        num_classes=dcfg["num_classes"],
    )
    return DataLoader(ds, batch_size=cfg["EVAL"]["batch_size"], shuffle=False,
                      num_workers=dcfg["num_workers"], pin_memory=True)


@torch.no_grad()
def predict(model, loader, device, amp_dtype, flip: bool = False):
    """返回 (概率 [N, C], 标签 [N])。"""
    probs, labels = [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        if flip:
            x = torch.flip(x, dims=[3])
        if amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(x)
        else:
            logits = model(x)
        probs.append(logits.float().softmax(dim=1).cpu())
        labels.append(batch["label"])
    return torch.cat(probs), torch.cat(labels)


def topk(probs: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    _, pred = probs.topk(5, dim=1)
    hit = pred.eq(labels.view(-1, 1))
    return (hit[:, :1].any(1).float().mean().item() * 100,
            hit.any(1).float().mean().item() * 100)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 每个视图 -> (crop_pct, 是否翻转)
    plan = {
        "base": (None, False),
        "flip": (None, True),
        "crop": (args.crop_pct, False),
        "zoom": (args.zoom_pct, False),
    }
    wanted = [v.strip() for v in args.views.split(",") if v.strip()]
    for name in wanted:
        if name not in plan:
            raise SystemExit(f"未知视图 {name}，可选 {list(plan)}")

    all_probs, labels = [], None
    for ci, path in enumerate(args.checkpoint, 1):
        ckpt = read_checkpoint(path, map_location=device)
        cfg = ckpt["config"]
        amp_dtype = resolve_amp(cfg)

        model = build_model(cfg).to(device)
        load_state_dict_tolerant(model, ckpt["model"])
        model.eval()

        # 同一个 checkpoint 的多视图融合
        per_ckpt = []
        for name in wanted:
            pct, flip = plan[name]
            loader = make_loader(cfg, model, pct, args.split)
            p, y = predict(model, loader, device, amp_dtype, flip=flip)
            labels = y
            per_ckpt.append(p)
            t1, t5 = topk(p, y)
            print(f"[{ci}] {path.split('/')[-1]:20s} {name:5s} "
                  f"flip={flip!s:5s} top1={t1:.2f} top5={t5:.2f}", flush=True)

        fused_ckpt = torch.stack(per_ckpt).mean(0)
        t1, t5 = topk(fused_ckpt, labels)
        print(f"[{ci}] 该权重的 {len(wanted)} 视图融合: top1={t1:.2f} top5={t5:.2f}\n",
              flush=True)
        all_probs.extend(per_ckpt)

    fused = torch.stack(all_probs).mean(0)
    t1, t5 = topk(fused, labels)
    print(f"[{args.split}] 总融合 {len(args.checkpoint)} 权重 × {len(wanted)} 视图 "
          f"= {len(all_probs)} 路: top1={t1:.2f}  top5={t5:.2f}")

    # 存逐样本概率：两个配置之间做配对检验（McNemar）必须用同一批测试图上的
    # 逐样本对错，不能只比 top1 —— 后者的标准误是 0.41 个点，比大部分差异还大。
    if args.dump:
        import numpy as np
        np.save(f"{args.dump}_probs.npy", fused.numpy())  # fp32：fp16 会改变个别近似并列样本的 argmax
        np.save(f"{args.dump}_labels.npy", labels.numpy())
        print(f"已保存逐样本概率: {args.dump}_probs.npy")


if __name__ == "__main__":
    main()
