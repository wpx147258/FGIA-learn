"""数据层。

对外只暴露 `build_dataloaders(cfg, backbone)`，train.py / test.py 不用关心
数据集怎么读、transform 怎么建。
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.cub import CUBDataset
from dataset.mixup import build_mixer
from dataset.sampler import PKBatchSampler
from dataset.transforms import build_transform, resolve_data_config
from utils.config import resolve_path
from utils.data_split import get_split_data
from utils.seeds import worker_init_fn


def build_dataloaders(cfg: dict, backbone,
                      splits=("train", "val", "test")) -> dict[str, DataLoader]:
    """构建 DataLoader 字典。

    参数：
        cfg:      合并后的完整配置
        backbone: 模型实例。**必须传**，因为归一化参数要从它的 pretrained_cfg 解析，
                  见 dataset/transforms.py 的说明。
        splits:   要构建哪些划分，test.py 可以只传 ("test",)

    返回：{"train": DataLoader, "val": DataLoader, ...}
    """
    dcfg = cfg["DATASET"]
    root: Path = resolve_path(dcfg["root"])
    num_workers: int = dcfg["num_workers"]

    # 归一化参数跟随 backbone，不手写
    data_cfg = dict(resolve_data_config(backbone))
    # 评测时的放大倍数默认跟随预训练配置（R50=0.875、CLIP-RN50=0.9、ViT=1.0）。
    # 换主干做对照时要统一成同一个口径，否则「涨点」里混着视野差异。
    if dcfg.get("crop_pct"):
        data_cfg["crop_pct"] = float(dcfg["crop_pct"])

    # 固定 DataLoader 的随机源，否则每个 epoch 的 shuffle 顺序不可复现
    generator = torch.Generator()
    generator.manual_seed(cfg["SEED"])

    # Mixup/CutMix 挂在 collate_fn 上，且**只给训练 loader**。
    # 两个 α 都是 0 时 build_mixer 返回 None，collate_fn=None 就是默认行为。
    mixer = build_mixer(cfg["TRAIN"], dcfg["num_classes"])

    # 三流融合（MODEL.fusion=head_body）需要额外裁出头部/身体两个局部。
    # 部位图用 mild=True 的 transform：裁剪本身已经收紧了画面，再叠常规的
    # RandomResizedCrop(0.5,1.0) 会把仅剩的上下文也裁掉。
    # 三流融合要用部位图；「GT 头框当一致性教师」只要 head 那一张，也走同一条链路
    use_parts = (cfg["MODEL"].get("fusion") == "head_body"
                 or cfg["TRAIN"].get("cam_teacher") == "gt_head")
    # 额外的部位视图（"wing,tail"）：训练期特权裁剪用，见 model/cam_aug.py
    extra_views = tuple(
        v.strip() for v in str(cfg["TRAIN"].get("cam_part_views") or "").split(",")
        if v.strip())
    if extra_views:
        use_parts = True
    # 额外部位视图的分辨率：更小以省显存（见 CUBDataset 的说明）
    part_size = int(cfg["TRAIN"].get("cam_part_size") or 224)

    # 成对判别要求批内有同类样本，随机采样给不了 —— 换成 P×K 批采样器。
    # TRAIN.pk_sampler 可以单独打开，用来做「只换采样、不加成对损失」的对照：
    # P×K 会让每批只覆盖 P 个类，这个变量本身就可能影响结果。
    use_pk = (bool(cfg["MODEL"].get("pair", False))
              or bool(cfg["TRAIN"].get("pk_sampler", False)))
    pair_p = int(cfg["TRAIN"].get("pair_p") or 8)
    pair_k = int(cfg["TRAIN"].get("pair_k") or 4)

    loaders = {}
    for split in splits:
        is_train = split == "train"
        tcfg = cfg["TRAIN"] if is_train else None
        dataset = CUBDataset(
            data_root=root,
            split=split,
            transform=build_transform(data_cfg, dcfg["image_size"], train=is_train,
                                      tcfg=tcfg),
            num_classes=dcfg["num_classes"],
            use_parts=use_parts,
            part_transform=build_transform(data_cfg, dcfg["image_size"],
                                           train=is_train, tcfg=tcfg, mild=True),
            # 只有训练需要这些额外视图；评估时算它们是纯浪费（测试不走辅助分支）
            extra_views=extra_views if is_train else (),
            part_transform_extra=(
                build_transform(data_cfg, part_size, train=is_train, tcfg=tcfg, mild=True)
                if (is_train and extra_views) else None),
        )
        # 把 val 并进训练：val 本来就是从官方 train 里切出来的，合回去就是完整的
        # 5994 张。只改 data_ids，路径/标签/部位框都是按 image_id 查表的，自动对得上。
        if is_train and dcfg.get("train_on_val", False):
            val_ids = get_split_data(root / "train_test_split.txt", "val")
            dataset.data_ids = list(dataset.data_ids) + list(val_ids)
            print(f"[data] train_on_val=True：训练集 {len(dataset.data_ids)} 张"
                  f"（val 已并入，val 指标从此不可信）")

        loader_kwargs = dict(
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            generator=generator,
            # 只在训练集上混合；评估必须确定性
            collate_fn=mixer if is_train else None,
        )
        if is_train and use_pk:
            # 标签从 dataset 的实际 data_ids 取，这样 train_on_val 合并后也对得上
            labels = [dataset.imgs_labels[i] for i in dataset.data_ids]
            sampler = PKBatchSampler(labels, pair_p, pair_k, seed=cfg["SEED"])
            loaders[split] = DataLoader(dataset, batch_sampler=sampler, **loader_kwargs)
            print(f"[data] 成对判别：P×K = {pair_p}×{pair_k}，每轮 "
                  f"{len(sampler)} 批（TRAIN.batch_size 不生效）")
        else:
            loaders[split] = DataLoader(
                dataset,
                batch_size=cfg["TRAIN" if is_train else "EVAL"]["batch_size"],
                shuffle=is_train,
                **loader_kwargs,
            )

    return loaders

