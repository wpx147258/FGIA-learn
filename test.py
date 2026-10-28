"""在测试集上评估已训练的 checkpoint。
用法：
    python test.py --checkpoint outputs/baseline_resnet50/20261014-233000/best.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dataset import build_dataloaders
from engine.build import build_criterion
from engine.trainer import evaluate, read_checkpoint, resolve_amp
from model.classifier import build_model
from utils.config import resolve_path


def parse_args():
    parser = argparse.ArgumentParser(description="CUB-200-2011 测试")
    parser.add_argument("--checkpoint", required=True, help="best.pth / last.pth 路径")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--worst-n", type=int, default=10,
                        help="额外打印准确率最低的 N 个类")
    return parser.parse_args()


def read_class_names(root) -> list[str]:
    """读 classes.txt（'<class_id> <类名>'，class_id 为 1..200），返回 0-based 的名字列表。"""
    names = {}
    with open(Path(root) / "classes.txt") as f:
        for line in f:
            line = line.strip()
            if line:
                class_id, name = line.split(maxsplit=1)
                names[int(class_id) - 1] = name
    return [names[i] for i in sorted(names)]


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = read_checkpoint(args.checkpoint, map_location=device)

    # 用训练时的配置重建模型 —— 这样 transform、num_classes、backbone
    # 都和当时完全一致，不会出现「权重能加载但结果对不上」。
    cfg = ckpt["config"]

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"checkpoint : {args.checkpoint}")
    print(f"epoch      : {ckpt['epoch']}    best val top1: {ckpt['best_metric']:.2f}")

    loaders = build_dataloaders(cfg, model.backbone, splits=(args.split,))
    loader = loaders[args.split]

    metrics = evaluate(
        model, loader, build_criterion(cfg), device,
        amp_dtype=resolve_amp(cfg), num_classes=cfg["DATASET"]["num_classes"],
    )

    n = len(loader.dataset)
    print(f"\n[{args.split}] n={n}  loss={metrics['loss']:.4f}  "
          f"top1={metrics['top1']:.2f}  top5={metrics['top5']:.2f}  "
          f"macro(top1)={metrics['macro_acc']:.2f}")

    # 每类准确率：细粒度任务上「哪些物种被混淆」比总分更有信息量
    acc = metrics["per_class_acc"]
    count = metrics["per_class_count"]
    names = read_class_names(resolve_path(cfg["DATASET"]["root"]))

    order = torch.argsort(torch.nan_to_num(acc, nan=1e9))[: args.worst_n]
    print(f"\n最差的 {args.worst_n} 个类：")
    for i in order.tolist():
        print(f"  {acc[i]:6.2f}%  n={int(count[i]):3d}  {names[i]}")

    out_path = Path(args.checkpoint).parent / f"{args.split}_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "epoch": ckpt["epoch"],
                "split": args.split,
                "n": int(count.sum()),
                "loss": metrics["loss"],
                "top1": metrics["top1"],
                "top5": metrics["top5"],
                "macro_acc": metrics["macro_acc"],
                "per_class_acc": [
                    None if torch.isnan(v) else round(float(v), 4) for v in acc
                ],
                "per_class_count": [int(v) for v in count],
                "class_names": names,
            },
            f, indent=2,
        )
    print(f"\n结果写入 {out_path}")


if __name__ == "__main__":
    main()
