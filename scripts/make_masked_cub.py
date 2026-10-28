"""用 CUB 自带的框生成「去背景」数据集，用来量模型的上限。

为什么要先量上限：CUB 的框平均只占原图 34.6%（中位 31.7%），
也就是说一张典型图片里 68% 的像素是背景。所以"按框裁出来再送网络"
实际上是两件事同时发生：

    1) 去掉了 68% 的干扰像素
    2) 鸟被放大约 1.8 倍，等效分辨率从 448 变成约 800

第二条单独看就是提分辨率。不拆开的话，测出来的"上限"主要来自放大，
不能说明干扰是不是瓶颈。所以这里提供两种模式：

    --mode mask   画布尺寸不变，框外填均值色      -> 只改干扰
    --mode crop   按框（外扩 pad 倍）裁出再保存   -> 分辨率和干扰一起变

用法：
    python scripts/make_masked_cub.py --mode mask --out data/CUB-200-2011-mask
    python scripts/make_masked_cub.py --mode crop --pad 1.1 --out data/CUB-200-2011-crop

生成出来的是一个和原数据集同结构的目录（images/ + 软链过来的 txt），
所以把配置里的 DATASET.root 指过去就行，dataset 代码一行不用改。

框的清洗：11788 个框里没有退化框，30 个越界、40 个覆盖超过 90%，
统一夹紧到图像边界即可，不丢样本。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# 从仓库根以外的 cwd 跑时也能 import（与 probe_pretrained.py 一致）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import resolve_path


def parse_args():
    p = argparse.ArgumentParser(description="生成去背景的 CUB 数据集")
    p.add_argument("--src", default="data/CUB-200-2011")
    p.add_argument("--out", required=True)
    p.add_argument("--mode", required=True, choices=["mask", "crop"])
    p.add_argument("--pad", type=float, default=1.1,
                   help="crop 模式下框的外扩倍数（mask 模式不用）")
    p.add_argument("--feather", type=int, default=3,
                   help="mask 模式下掩码边缘羽化的像素数，避免硬边成为捷径")
    p.add_argument("--fill", default="mean", choices=["mean", "global"],
                   help="mean = 每张图自己的均值色；global = 全体均值色")
    p.add_argument("--quality", type=int, default=95, help="JPEG 质量")
    p.add_argument("--limit", type=int, default=0,
                   help="只处理前 N 张，用来冒烟（0 = 全部）")
    return p.parse_args()


def clamp_box(x, y, w, h, W, H):
    """夹到图像内，并保证宽高至少 4 像素。"""
    x0 = max(0.0, min(x, W - 4.0))
    y0 = max(0.0, min(y, H - 4.0))
    x1 = min(float(W), max(x0 + 4.0, x + w))
    y1 = min(float(H), max(y0 + 4.0, y + h))
    return x0, y0, x1, y1


def main():
    args = parse_args()
    src = resolve_path(args.src)
    out = resolve_path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    # txt 全部软链过去：划分、标签、类名都按 image_id 查表，路径变了也能对上
    for name in ("images.txt", "image_class_labels.txt", "train_test_split.txt",
                 "classes.txt", "bounding_boxes.txt"):
        dst = out / name
        if not dst.exists():
            dst.symlink_to((src / name).resolve())
    if (src / "parts").exists() and not (out / "parts").exists():
        (out / "parts").symlink_to((src / "parts").resolve())

    paths = {}
    with open(src / "images.txt") as f:
        for line in f:
            a = line.split()
            if a:
                paths[int(a[0])] = a[1]

    boxes = {}
    with open(src / "bounding_boxes.txt") as f:
        for line in f:
            a = line.split()
            if a:
                boxes[int(a[0])] = tuple(map(float, a[1:5]))

    # global 填充色：在采样上算一次全体均值
    global_fill = None
    if args.fill == "global":
        acc, n = np.zeros(3), 0
        for i in list(boxes)[:300]:
            arr = np.asarray(Image.open(src / "images" / paths[i]).convert("RGB"))
            acc += arr.reshape(-1, 3).mean(0) * arr.shape[0] * arr.shape[1]
            n += arr.shape[0] * arr.shape[1]
        global_fill = (acc / n).round().astype(np.uint8)

    ids = sorted(boxes)
    if args.limit:
        ids = ids[: args.limit]

    for k, img_id in enumerate(ids, 1):
        img_path = src / "images" / paths[img_id]
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        x, y, w, h = boxes[img_id]
        x0, y0, x1, y1 = clamp_box(x, y, w, h, W, H)

        if args.mode == "crop":
            # 以框中心为基准外扩，再夹回图像
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            hw, hh = (x1 - x0) * args.pad / 2, (y1 - y0) * args.pad / 2
            bx0, by0 = max(0, int(round(cx - hw))), max(0, int(round(cy - hh)))
            bx1, by1 = min(W, int(round(cx + hw))), min(H, int(round(cy + hh)))
            new_im = im.crop((bx0, by0, bx1, by1))
        else:
            arr = np.asarray(im).astype(np.float32)
            mask = np.zeros((H, W), dtype=np.float32)
            mask[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = 1.0
            if args.feather > 0:
                # 羽化只是为了不让掩码边界变成一条锐利的直线（那会变成捷径）
                blur = Image.fromarray((mask * 255).astype(np.uint8)).filter(
                    ImageFilter.GaussianBlur(args.feather))
                mask = np.asarray(blur).astype(np.float32) / 255.0
            fill = global_fill if global_fill is not None else arr.reshape(-1, 3).mean(0)
            m3 = mask[..., None]
            arr = arr * m3 + fill.astype(np.float32) * (1 - m3)
            new_im = Image.fromarray(arr.round().clip(0, 255).astype(np.uint8))

        dst = out / "images" / paths[img_id]
        dst.parent.mkdir(parents=True, exist_ok=True)
        new_im.save(dst, quality=args.quality)
        if k % 1000 == 0:
            print(f"  {k}/{len(boxes)}", flush=True)

    print(f"完成：{out}  模式={args.mode}  图片 {len(ids)} 张")


if __name__ == "__main__":
    main()
