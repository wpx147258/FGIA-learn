"""从 metrics.csv 画训练曲线。

用法：
    python scripts/plot_curves.py <run_dir> [...]      # 指定若干 run
    python scripts/plot_curves.py --all                # 扫描 outputs/ 下所有 run
    python scripts/plot_curves.py --all --force        # 连已有的也重画

输出写到每个 run 目录下的 curves.png。train.py 训练结束后会自动调用一次。

只依赖 metrics.csv，所以历史 run 都能补画——不用重新训练。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def read_metrics(run_dir: Path) -> dict[str, list[float]]:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return {}
    cols: dict[str, list[float]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                try:
                    cols.setdefault(k, []).append(float(v))
                except (TypeError, ValueError):
                    pass
    return cols


def plot(run_dir: Path, force: bool = False) -> Path | None:
    out = run_dir / "curves.png"
    if out.exists() and not force:
        return None
    d = read_metrics(run_dir)
    if not d or "epoch" not in d:
        return None

    ep = d["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0, 0]
    for k, c in (("train_loss", "C0"), ("val_loss", "C1")):
        if k in d:
            ax.plot(ep, d[k], c, label=k, lw=1.4)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(fontsize=8)
    ax.set_title("loss"); ax.grid(alpha=.25)

    ax = axes[0, 1]
    for k, c in (("val_top1", "C1"), ("val_top5", "C2")):
        if k in d:
            ax.plot(ep, d[k], c, label=k, lw=1.4)
    ax.set_xlabel("epoch"); ax.set_ylabel("%"); ax.legend(fontsize=8)
    ax.set_title("val accuracy"); ax.grid(alpha=.25)

    ax = axes[1, 0]
    if "lr" in d:
        ax.plot(ep, d["lr"], "C3", lw=1.4)
        ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("lr (log)"); ax.set_title("learning rate")
    ax.grid(alpha=.25)

    ax = axes[1, 1]
    for k, c in (("train_top1", "C0"),):
        if k in d:
            ax.plot(ep, d[k], c, label=k, lw=1.2)
    ax.set_xlabel("epoch"); ax.set_ylabel("%"); ax.legend(fontsize=8)
    ax.set_title("train accuracy (meaningless under mixup)")
    ax.grid(alpha=.25)

    fig.suptitle(run_dir.name, fontsize=10)
    plt.tight_layout()
    plt.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return out


def find_runs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob("metrics.csv"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true", help="扫描 outputs/ 下所有 run")
    ap.add_argument("--force", action="store_true", help="已有的 curves.png 也重画")
    args = ap.parse_args()

    dirs = []
    if args.all:
        dirs += find_runs(ROOT / "outputs")
    dirs += [p if p.is_absolute() else (ROOT / p) for p in args.runs]

    made = skipped = 0
    for d in dirs:
        if not d.is_dir():
            continue
        if plot(d, args.force):
            made += 1
        else:
            skipped += 1
    print(f"绘制 {made} 个，跳过 {skipped} 个（已存在或无数据）")


if __name__ == "__main__":
    main()
