"""CUB-200-2011 Dataset。

`use_parts=True` 时额外读 `parts/part_locs.txt`，为每张图裁出「头部」和「身体」
两个局部，随样本一起返回（`head` / `body` 两个键）。这是三流融合的数据来源：
    整图 + 头部 + 身体 → 共享 backbone → 特征拼接 → 分类
关键点用它是因为 CUB 的头部部位几乎 100% 可见（beak 99.6%、throat 98.7%、
forehead 98.4%、crown 98.2%、nape 96.0%），而这类局部正是区分近缘鸟种的依据。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from utils.data_split import get_split_data


# parts.txt 的编号（1..15）
HEAD_PARTS = (2, 5, 6, 7, 10, 11, 15)      # beak, crown, forehead, eyes, nape, throat
BODY_PARTS = (1, 3, 4, 8, 9, 12, 13, 14)   # back, belly, breast, legs, wings, tail


def box_from_points(points, W: int, H: int,
                    pad: float = 0.35, min_side: int = 48):
    """一组关键点 -> 外接矩形 (x0, y0, x1, y1)，像素坐标。

    pad 是**每边**外扩的比例（0.35 = 长宽各放大 70%），留出上下文；
    min_side 防止「只有一两个点可见」时裁出一个小方块再被放大成噪声。
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w = max(x1 - x0, min_side)
    h = max(y1 - y0, min_side)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w, half_h = w * (0.5 + pad), h * (0.5 + pad)
    return (
        max(0, int(cx - half_w)), max(0, int(cy - half_h)),
        min(W, int(cx + half_w)), min(H, int(cy + half_h)),
    )


class CUBDataset(Dataset):

    def __init__(self, data_root, split: str, transform, num_classes: int,
                 use_parts: bool = False, part_transform=None,
                 part_pad: float = 0.35):
        super().__init__()
        self.root = Path(data_root)  # ./data/CUB-200-2011
        self.transform = transform
        self.num_classes = num_classes
        self.split = split
        self.data_ids = get_split_data(self.root / "train_test_split.txt", split)
        self.imgs_paths = self.get_paths(self.root / "images.txt")
        self.imgs_labels = self.get_labels(self.root / "image_class_labels.txt")

        self.use_parts = use_parts
        self.part_transform = part_transform or transform
        self.part_pad = part_pad
        if use_parts:
            self.parts = self.get_parts(self.root / "parts" / "part_locs.txt")
            self.bboxes = self.get_bboxes(self.root / "bounding_boxes.txt")

    def get_paths(self, images_txt: Path) -> dict[int, Path]:
        paths = {}
        with open(images_txt) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_id, img_name = line.split()
                paths[int(img_id)] = self.root / "images" / img_name
        return paths

    def get_labels(self, labels_txt: Path) -> dict[int, int]:
        labels = {}
        with open(labels_txt) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_id, class_id = line.split()
                labels[int(img_id)] = int(class_id) - 1
        return labels

    @staticmethod
    def get_parts(path: Path) -> dict[int, dict[int, tuple[float, float, int]]]:
        """{image_id: {part_id: (x, y, visible)}}"""
        out: dict[int, dict[int, tuple[float, float, int]]] = {}
        with open(path) as f:
            for line in f:
                iid, pid, x, y, v = line.split()
                out.setdefault(int(iid), {})[int(pid)] = (float(x), float(y), int(v))
        return out

    @staticmethod
    def get_bboxes(path: Path) -> dict[int, tuple[float, float, float, float]]:
        """{image_id: (x, y, w, h)}，像素坐标。"""
        out = {}
        with open(path) as f:
            for line in f:
                iid, x, y, w, h = line.split()
                out[int(iid)] = (float(x), float(y), float(w), float(h))
        return out

    def _part_boxes(self, image_id: int, W: int, H: int):
        locs = self.parts.get(image_id, {})

        def visible(ids):
            return [(locs[i][0], locs[i][1]) for i in ids
                    if i in locs and locs[i][2] == 1]

        head, body = visible(HEAD_PARTS), visible(BODY_PARTS)
        centre = [(W / 2, H / 2)]
        head_box = box_from_points(head or centre, W, H, self.part_pad)
        body_box = box_from_points(body or centre, W, H, self.part_pad)

        # 体部关键点有时会挤在一小片（侧面站姿、遮挡），外接框小到只剩喙的一截。
        # 实测 24.8% 的图体部框 < 100x100。这种情况下退回整只鸟的 bbox，
        # 让 body 分支至少看到鸟的完整轮廓，而不是一块纹理。
        bb = self.bboxes.get(image_id)
        if bb is not None:
            bx, by, bw, bh = bb
            bbox_area = max(bw * bh, 1.0)
            body_area = (body_box[2] - body_box[0]) * (body_box[3] - body_box[1])
            if body_area < 0.30 * bbox_area:
                body_box = (max(0, int(bx)), max(0, int(by)),
                            min(W, int(bx + bw)), min(H, int(by + bh)))
        return head_box, body_box

    def __len__(self) -> int:
        return len(self.data_ids)

    def __getitem__(self, idx: int) -> dict:
        """返回 {"image", "label", "image_id"}；use_parts 时另加 {"head", "body"}。"""
        image_id = self.data_ids[idx]
        label = self.imgs_labels[image_id]
        path = self.imgs_paths[image_id]

        with Image.open(path) as img:
            img = img.convert("RGB")
            sample = {
                "image": self.transform(img),
                "label": label,
                "image_id": image_id,
            }
            if self.use_parts:
                W, H = img.size
                head_box, body_box = self._part_boxes(image_id, W, H)
                sample["head"] = self.part_transform(img.crop(head_box))
                sample["body"] = self.part_transform(img.crop(body_box))
        return sample
