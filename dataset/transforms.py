"""图像预处理。

归一化参数**不写死**，从 backbone 的 pretrained_cfg 解析，换 backbone 自动跟随。
手写 mean/std 一旦和预训练权重不匹配，表现是「训得动但就是上不去」，
不会报错，很难查。

⚠️ 必须传**模型实例**而不是模型名字符串：timm 对同一架构，`resolve_data_config`
传实例和传名字可能给出不同的 crop_pct（实测 resnet50：实例 0.95，字符串 0.875）。
"""

from __future__ import annotations

import timm
from timm.data.auto_augment import rand_augment_transform
from torchvision import transforms

# 细粒度分类的训练增强要比 ImageNet 保守。
# RandomResizedCrop 默认 scale=(0.08, 1.0)，会把鸟裁掉大半只剩背景，
# 在 CUB 上直接伤准确率，所以收紧到 (0.5, 1.0)。
# 可以被配置项 TRAIN.crop_scale 覆盖（见 build_transform），这里只是默认值。
TRAIN_CROP_SCALE = (0.5, 1.0)
TRAIN_CROP_RATIO = (3 / 4, 4 / 3)

# timm 的 RandAugment 一共 15 个操作，其中 10 个是颜色类
# （Color / Brightness / Contrast / Sharpness / Solarize / Posterize /
#   Equalize / AutoContrast / Invert / SolarizeAdd），5 个是几何类。
#
# CUB 上这个分界是要紧的：鸟的羽色**就是**判别特征 —— 最难的几个类
# （California_Gull 20%、Herring_Gull 50%、Common_Tern 37%）正是靠喙和腿的
# 颜色区分的，灰度下几乎无法分辨。所以颜色抖动在 ImageNet 上安全，
# 在这里有反伤风险。默认只保留几何操作；要开颜色用
# rand_augment_geometric_only=false 显式打开，把它当成一个**独立的实验变量**。
RANDAUG_GEOMETRIC_OPS = ["Rotate", "ShearX", "ShearY", "TranslateXRel", "TranslateYRel"]


def build_train_augment(tcfg: dict | None) -> list:
    """RandomResizedCrop 之后、ToTensor 之前插入的增强（都作用于 PIL 图像）。"""
    if not tcfg:
        return []
    cfg_str = tcfg.get("rand_augment") or ""
    if not cfg_str:
        return []
    only_geo = tcfg.get("rand_augment_geometric_only", True)
    return [rand_augment_transform(
        cfg_str, {}, transforms=RANDAUG_GEOMETRIC_OPS if only_geo else None,
    )]


def build_random_erasing(tcfg: dict | None) -> list:
    """ToTensor + Normalize 之后插入的增强（作用于张量）。"""
    if not tcfg:
        return []
    p = float(tcfg.get("random_erasing") or 0.0)
    if p <= 0:
        return []
    # value='random' 而不是 0：归一化之后填 0 相当于填一个固定的灰值，
    # 模型可能把这个恒定色块本身当成线索。填随机噪声没有这个问题。
    return [transforms.RandomErasing(p=p, value="random")]


def resolve_data_config(backbone) -> dict:
    """从 backbone 实例解析 {input_size, mean, std, crop_pct, interpolation}。"""
    return timm.data.resolve_data_config({}, model=backbone)


def build_transform(data_cfg: dict, image_size: int, train: bool,
                    tcfg: dict | None = None, mild: bool = False) -> transforms.Compose:
    """构建 train / eval 两套 transform。

    eval 的流程是 Resize(image_size / crop_pct) 再 CenterCrop(image_size)，
    和预训练时的评估流程保持一致 —— 这也是 timm 推荐的做法。

    tcfg 只在 train=True 时用（增强参数在 cfg['TRAIN'] 里）；eval 必须是确定性的。

    mild=True 给**已经裁好的部位图**用：部位裁剪本身已经把画面收紧了，
    再叠 (0.5,1.0) 的 RandomResizedCrop 会把仅剩的上下文也裁掉。
    这时只留很轻的随机裁剪（0.85~1.0），避免部位 patch 变成纯纹理。
    """
    mean, std = data_cfg["mean"], data_cfg["std"]
    # timm 里把字符串（'bicubic'）转成 torchvision InterpolationMode 的函数
    interpolation = timm.data.str_to_interp_mode(data_cfg.get("interpolation", "bicubic"))
    crop_pct = data_cfg.get("crop_pct") or 1.0
    resize_size = int(round(image_size / crop_pct))

    if train:
        # mild 是给已经裁好的部位图用的，固定保持轻裁剪；
        # 主图允许用 TRAIN.crop_scale 覆盖（比如给小目标更多放大）
        if mild:
            scale = (0.85, 1.0)
        else:
            scale = tuple((tcfg or {}).get("crop_scale") or TRAIN_CROP_SCALE)
        return transforms.Compose([
            transforms.RandomResizedCrop(
                image_size, scale=scale, ratio=TRAIN_CROP_RATIO,
                interpolation=interpolation,
            ),
            transforms.RandomHorizontalFlip(),
            *build_train_augment(tcfg),      # PIL 阶段：RandAugment（默认仅几何）
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            *build_random_erasing(tcfg),     # 张量阶段：Random Erasing
        ])

    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

if __name__ == "__main__":
    backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
    data_cfg = timm.data.resolve_data_config({}, model=backbone)
    print(data_cfg)