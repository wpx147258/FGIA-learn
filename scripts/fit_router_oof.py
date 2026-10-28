"""用 OOF 数据拟合「多区域 → 主分支」的切换门控（learning-to-defer 的落地版）。

为什么必须 OOF：现有 val 只有 599 张、且「区域一致且不同」的样本极少（个位数），
在其上选阈值等于在噪声上选；而直接在 test 上选阈值会被指为选择性偏倚。
这里冻结已训练好的模型，在**训练集**上做 5 折 OOF：用 4 折拟合 gate、预测第 5 折，
最后在 test 上只评估一次。

流程：
  1. 前向收集 train/test 的 main_logits、region_logits、routing_features、labels；
  2. 5 折：拟合两个 logistic 模型 p_main(x)、p_region(x)（特征相同）；
  3. 切换规则：p_region(x) - p_main(x) > τ，τ 在 OOF 上按 precision-coverage 选；
  4. test 上报告：主分支、区域多数票、OOF-τ 切换后的准确率与触发数。

用法：
  python scripts/fit_router_oof.py --checkpoint outputs/r50_region_route/<ts>/swa_f0.5.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.cub import CUBDataset
from dataset.transforms import build_transform, resolve_data_config
from engine.trainer import read_checkpoint, resolve_amp
from model.classifier import build_model
from model.regions import routing_features
from utils.config import resolve_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--dump", default="")
    return p.parse_args()


@torch.no_grad()
def collect(model, loader, device, amp):
    mains, regions, feats, ys = [], [], [], []
    for b in loader:
        x = b["image"].to(device)
        out = model._forward_region_route(x)          # eval 路径返回 fused；这里手动取中间量
        # 重新走一遍以拿到各分量（eval 模式下 _forward_region_route 直接返回 fused）
        feat = model.backbone.forward_features(x)
        pooled = model.backbone.forward_head(feat)
        logits = model.head(pooled)
        fm = model._to_feature_map(feat)
        topk = logits.float().topk(model.rr_regions, dim=1).indices
        w = model.head.cam_weight(fm.shape[1]).detach()
        from model.cam_aug import cam_map, crop_and_resize
        rl, rf = [], []
        for i in range(model.rr_regions):
            cam = cam_map(fm.float(), w, topk[:, i])
            from model.regions import propose_regions
            boxes = propose_regions(cam, model.rr_alpha)
            patch = crop_and_resize(x, boxes, model.rr_size, model.rr_margin)
            f = model.backbone.forward_features(patch)
            rf.append(model.backbone.forward_head(f)); rl.append(model.head(rf[-1]))
        rl = torch.stack(rl, 1); rf = torch.stack(rf, 1)
        mains.append(logits.float().cpu()); regions.append(rl.float().cpu())
        feats.append(routing_features(logits, rl, pooled, rf).float().cpu())
        ys.append(b["label"])
    return (torch.cat(mains), torch.cat(regions), torch.cat(feats), torch.cat(ys))


def main():
    args = parse_args()
    from sklearn.linear_model import LogisticRegression
    dev = torch.device("cuda")
    ck = read_checkpoint(args.checkpoint, map_location=dev)
    cfg = ck["config"]; amp = resolve_amp(cfg)
    model = build_model(cfg).to(dev); model.load_state_dict(ck["model"]); model.eval()
    dcfg = cfg["DATASET"]
    dc = dict(resolve_data_config(model.backbone))
    if dcfg.get("crop_pct"): dc["crop_pct"] = float(dcfg["crop_pct"])
    tf = build_transform(dc, dcfg["image_size"], train=False)

    def loader(split):
        ds = CUBDataset(resolve_path(dcfg["root"]), split, tf, dcfg["num_classes"])
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=dcfg["num_workers"])

    Mtr, Rtr, Ftr, ytr = collect(model, loader("train"), dev, amp)
    Mte, Rte, Fte, yte = collect(model, loader("test"), dev, amp)
    if args.dump:
        np.savez(args.dump, Mtr=Mtr.numpy(), Rtr=Rtr.numpy(), Ftr=Ftr.numpy(), ytr=ytr.numpy(),
                 Mte=Mte.numpy(), Rte=Rte.numpy(), Fte=Fte.numpy(), yte=yte.numpy())

    def vote_of(R):
        return R.argmax(-1).mode(dim=1).values

    # 5 折 OOF：拟合 p_main 与 p_region
    n = len(ytr); idx = np.arange(n); rng = np.random.RandomState(0); rng.shuffle(idx)
    folds = np.array_split(idx, args.folds)
    oof_m = np.zeros(n); oof_r = np.zeros(n)
    Xtr = Ftr.numpy(); ytr_np = ytr.numpy()
    vtr = vote_of(Rtr).numpy()
    for f in folds:
        tr = np.setdiff1d(idx, f)
        cm = LogisticRegression(max_iter=2000).fit(Xtr[tr], (Mtr.argmax(1).numpy()[tr] == ytr_np[tr]).astype(int))
        cr = LogisticRegression(max_iter=2000).fit(Xtr[tr], (vtr[tr] == ytr_np[tr]).astype(int))
        oof_m[f] = cm.predict_proba(Xtr[f])[:, 1]
        oof_r[f] = cr.predict_proba(Xtr[f])[:, 1]
    # 在 OOF 上选 τ（要求切换的样本数 >= 1%）
    best = None
    for tau in np.linspace(-0.2, 0.4, 61):
        use = (oof_r - oof_m) > tau
        if use.sum() < 0.005 * n: continue
        pred = Mtr.argmax(1).numpy().copy(); pred[use] = vtr[use]
        acc = (pred == ytr_np).mean()
        if best is None or acc > best[0]:
            best = (acc, tau, int(use.sum()))
    acc_oof, tau, n_use = best
    print(f"OOF 选定 τ={tau:.3f}（触发 {n_use}/{n}）；OOF 准确率 {acc_oof*100:.2f}%")
    # test 上只评估一次
    cm = LogisticRegression(max_iter=2000).fit(Xtr, (Mtr.argmax(1).numpy() == ytr_np).astype(int))
    cr = LogisticRegression(max_iter=2000).fit(Xtr, (vtr == ytr_np).astype(int))
    Fte_np = Fte.numpy(); yte_np = yte.numpy()
    p_m = cm.predict_proba(Fte_np)[:, 1]; p_r = cr.predict_proba(Fte_np)[:, 1]
    vte = vote_of(Rte).numpy(); main_pred = Mte.argmax(1).numpy()
    use = (p_r - p_m) > tau
    pred = main_pred.copy(); pred[use] = vte[use]
    print(f"test: 主分支 {(main_pred==yte_np).mean()*100:.2f}  切换后 {(pred==yte_np).mean()*100:.2f} "
          f"(触发 {int(use.sum())}: 改对 {int(((pred==yte_np)&(main_pred!=yte_np)).sum())} "
          f"改错 {int(((pred!=yte_np)&(main_pred==yte_np)).sum())})")


if __name__ == "__main__":
    main()
