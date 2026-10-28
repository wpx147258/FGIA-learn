"""离线分析：部件的独立预测能不能用来「门控」主分支——只在可靠时采纳部件意见。

pa4 相对基线修好 109 张、弄坏 124 张（净 −15）。如果能从输出里识别出「该信任部件」
的样本，这 109 张就可能变成净收益。这里把主 logits 与 4 个部件的 logits 都 dump 出来，
在 test 上扫几种无需训练的门控规则。

用法：python scripts/probe_part_gate.py --checkpoint <pa4 ckpt>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.cub import CUBDataset
from dataset.transforms import build_transform, resolve_data_config
from engine.trainer import read_checkpoint, resolve_amp
from model.classifier import build_model
from utils.config import resolve_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--dump", default="")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    dev = torch.device("cuda")
    ck = read_checkpoint(args.checkpoint, map_location=dev)
    cfg = ck["config"]; amp = resolve_amp(cfg)
    model = build_model(cfg).to(dev); model.load_state_dict(ck["model"]); model.eval()
    dcfg = cfg["DATASET"]
    data_cfg = dict(resolve_data_config(model.backbone))
    if dcfg.get("crop_pct"):
        data_cfg["crop_pct"] = float(dcfg["crop_pct"])
    ds = CUBDataset(resolve_path(dcfg["root"]), args.split,
                    build_transform(data_cfg, dcfg["image_size"], train=False),
                    dcfg["num_classes"])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=dcfg["num_workers"], pin_memory=True)
    pa = model.backbone.part_attn
    mains, parts, ys = [], [], []
    for batch in loader:
        x = batch["image"].to(dev); ys.append(batch["label"])
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            f = model.backbone.forward_features(x)
            lg = model.head(model.backbone.forward_head(f))
        lp = pa._last_part_logits
        mains.append(lg.float().cpu()); parts.append(lp.float().cpu())
    main_l = torch.cat(mains); part_l = torch.cat(parts); y = torch.cat(ys)
    if args.dump:
        import numpy as np
        np.save(f"{args.dump}_main.npy", main_l.numpy())
        np.save(f"{args.dump}_parts.npy", part_l.numpy())
        np.save(f"{args.dump}_labels.npy", y.numpy())

    pm = main_l.softmax(1); pp = part_l.softmax(-1)              # [N,C], [N,K,C]
    pred_m = pm.argmax(1)
    pred_p = pp.argmax(-1)                                        # [N,K]
    acc = lambda p: float((p == y).float().mean() * 100)
    print(f"main      {acc(pred_m):.2f}")
    print(f"part1..K  " + "  ".join(f"{acc(pred_p[:,k]):.2f}" for k in range(pred_p.shape[1])))
    # 部件一致票
    k = pred_p.shape[1]
    vote = torch.mode(pred_p, dim=1).values
    agree = (pred_p == vote[:, None]).sum(1)
    print(f"部件多数票 {acc(vote):.2f}（全一致比例 {(agree==k).float().mean()*100:.1f}%）")
    # 门控：部件一致且与主分支不同的样本，改判
    for need in range(2, k + 1):
        use = (agree >= need) & (vote != pred_m)
        pred = pred_m.clone(); pred[use] = vote[use]
        ok_new = (pred == y); ok_old = (pred_m == y)
        fixed = int((ok_new & ~ok_old).sum()); broke = int((~ok_new & ok_old).sum())
        print(f"若 ≥{need}/{k} 部件一致且不同于主分支则改判: {float(ok_new.float().mean()*100):.2f} "
              f"(触发 {int(use.sum())}: 改对 {fixed} 改错 {broke})")
    # 门控：只在共识类位于主分支 top-m 时才改判（避免采纳完全无关的类）
    for m in (2, 3, 5):
        topm = pm.topk(m, dim=1).indices
        in_top = (topm == vote[:, None]).any(1)
        for need in (k - 1, k):
            use = (agree >= need) & (vote != pred_m) & in_top
            pred = pred_m.clone(); pred[use] = vote[use]
            ok_new = (pred == y); ok_old = (pred_m == y)
            print(f"≥{need}/{k} 一致 + 共识类在主分支 top-{m}: {float(ok_new.float().mean()*100):.2f} "
                  f"(触发 {int(use.sum())}: 改对 {int((ok_new&~ok_old).sum())} 改错 {int((~ok_new&ok_old).sum())})")
    # 门控：部件一致票的置信度高于主分支才改判
    conf_v = pp.gather(2, vote[:, None, None].expand(-1, k, 1)).mean(1).squeeze(-1)
    conf_m = pm.max(1).values
    for th in (0.0, 0.02, 0.05, 0.1):
        use = (conf_v - conf_m > th) & (vote != pred_m) & (agree == k)
        pred = pred_m.clone(); pred[use] = vote[use]
        ok_new = (pred == y); ok_old = (pred_m == y)
        print(f"一致票置信度高出主分支 >{th:.2f}: {float(ok_new.float().mean()*100):.2f} "
              f"(触发 {int(use.sum())}: 改对 {int((ok_new&~ok_old).sum())} 改错 {int((~ok_new&ok_old).sum())})")


if __name__ == "__main__":
    main()
