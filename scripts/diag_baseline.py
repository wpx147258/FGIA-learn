"""对比当前两个基线的细粒度能力，输出可操作的诊断指标。

对比对象：
  a1      outputs/05_cam/a1_448_camcrop_w02/*/swa.pth        test 87.84
  swsl    outputs/swsl_448_alpha02/*/swa_f0.5.pth            test 87.65

指标：
  1. 粗组内 / 跨组错误占比（粗组 = 类名最后一个词，如 Gull / Tern / Warbler）
  2. 每类准确率 vs 该类训练样本数
  3. 特征空间的类内 / 类间余弦距离
  4. CAM 落在 GT 框内的质量比、峰值是否在框内（200 张抽样）
  5. 两个模型错误集合的重合度（判断集成空间）
"""

from __future__ import annotations

import glob
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import timm

from dataset import build_dataloaders
from model.cam_aug import mask_to_box, peak_mask
from model.classifier import Classifier
from utils.config import BASE_CONFIG, load_config, resolve_path

ROOT = Path(__file__).resolve().parents[1]
DEV = torch.device("cuda")


def load_ckpt(pat: str):
    p = sorted(glob.glob(str(ROOT / pat)))[0]
    ck = torch.load(p, map_location="cpu", weights_only=True)
    sd = ck["model"]
    bb = timm.create_model("resnet50", pretrained=False, num_classes=0)
    bb.load_state_dict({k[len("backbone."):]: v for k, v in sd.items()
                        if k.startswith("backbone.")}, strict=True)
    head = torch.nn.Linear(sd["head.fc.weight"].shape[1], sd["head.fc.weight"].shape[0])
    head.load_state_dict({"weight": sd["head.fc.weight"], "bias": sd["head.fc.bias"]})
    m = Classifier(bb, _Wrap(head)).to(DEV).eval()
    return m, head


class _Wrap(torch.nn.Module):
    """让 head 走 on_feature_map=False 的默认路径（输入是池化后的 [B,C]）。"""

    def __init__(self, fc):
        super().__init__()
        self.fc = fc

    def forward(self, x):
        return self.fc(x)


@torch.no_grad()
def run(model, loader):
    logits, feats, labels, ids = [], [], [], []
    for b in loader:
        x = b["image"].to(DEV, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            fmap = model.backbone.forward_features(x)
            pooled = model.backbone.forward_head(fmap)
            out = model.head(pooled)
        logits.append(out.float().cpu())
        feats.append(pooled.float().cpu())
        labels.append(b["label"])
        ids.append(b["image_id"])
    return torch.cat(logits), torch.cat(feats), torch.cat(labels), torch.cat(ids)


def main() -> None:
    classes = {}
    for line in open(resolve_path("data/CUB-200-2011/classes.txt")):
        i, name = line.split()
        classes[int(i) - 1] = name.split(".")[1]
    # 粗组 = 类名最后一个词（Albatross / Gull / Tern / Warbler ...）
    group = {k: v.split("_")[-1] for k, v in classes.items()}

    cfg = load_config([BASE_CONFIG, ROOT / "configs/baseline_config.yaml"],
                      ["DATASET.image_size=448", "DATASET.num_workers=8",
                       "EVAL.batch_size=32"])
    dummy = timm.create_model("resnet50", pretrained=False, num_classes=0)
    loaders = build_dataloaders(cfg, dummy, splits=("test", "train"))
    train_labels = [loaders["train"].dataset.imgs_labels[i]
                    for i in loaders["train"].dataset.data_ids]
    per_class_train = Counter(train_labels)

    res = {}
    for tag, pat in [("a1", "outputs/05_cam/a1_448_camcrop_w02/*/swa.pth"),
                     ("swsl", "outputs/swsl_448_alpha02/*/swa_f0.5.pth")]:
        m, _ = load_ckpt(pat)
        lg, ft, lb, ids = run(m, loaders["test"])
        pred = lg.argmax(1)
        acc = (pred == lb).float().mean().item() * 100
        wrong = pred != lb

        # ① 组内 vs 跨组
        same = torch.tensor([group[int(a)] == group[int(b)]
                             for a, b in zip(lb, pred)])
        within = (wrong & same).sum().item()
        print(f"\n=== {tag}  test top1={acc:.2f}  错误 {wrong.sum().item()}/{len(lb)} ===")
        print(f"  组内错误 {within} ({100*within/max(wrong.sum().item(),1):.0f}% of errors)  "
              f"跨组 {wrong.sum().item()-within}")

        # ② 每类准确率 vs 训练样本数
        pc = torch.zeros(200); pn = torch.zeros(200)
        for c, ok in zip(lb, ~wrong):
            pc[c] += ok; pn[c] += 1
        pa = (pc / pn.clamp(min=1)) * 100
        n_tr = torch.tensor([per_class_train.get(int(c), 0) for c in range(200)]).float()
        print(f"  每类准确率: 均值 {pa.mean():.1f}  最差 10 类 {pa.topk(10, largest=False).values.tolist()}")
        print(f"  最差 10 类名的粗组: {[group[int(i)] for i in pa.topk(10, largest=False).indices]}")

        # ③ 特征空间的类内 / 类间余弦
        fn = F.normalize(ft, dim=1)
        cent = torch.zeros(200, ft.shape[1])
        for c in range(200):
            sel = lb == c
            if sel.any():
                cent[c] = fn[sel].mean(0)
        cent = F.normalize(cent, dim=1)
        intra = (fn * cent[lb]).sum(1).mean().item()
        sim = cent @ cent.t()
        off = sim[~torch.eye(200, dtype=bool)]
        print(f"  特征余弦: 类内 {intra:.3f}   类间(中心-中心) 均值 {off.mean():.3f}  "
              f"最大 {off.max():.3f}")

        res[tag] = dict(pred=pred, label=lb, acc=acc, wrong=wrong)

    # ⑤ 两个模型的错误重合
    a, s = res["a1"]["wrong"], res["swsl"]["wrong"]
    both = (a & s).sum().item()
    either = (a | s).sum().item()
    print(f"\n=== 错误集合 ===")
    print(f"  a1 错 {a.sum().item()}  swsl 错 {s.sum().item()}  两者都错 {both}  任一错 {either}")
    print(f"  错误重合度 Jaccard = {both/max(either,1):.3f}  "
          f"（越低说明两个模型互补性越强）")
    print(f"  理论上限（取并集修正）= {(len(a)-both)/len(a)*100:.2f}% 提升空间")


if __name__ == "__main__":
    main()
