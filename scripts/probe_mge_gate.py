"""多粒度专家（MGE）的离线门控分析：专家一致票能否否决主分支。

用法：python scripts/probe_mge_gate.py --checkpoint <mge ckpt>
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.cub import CUBDataset
from dataset.transforms import build_transform, resolve_data_config
from engine.trainer import read_checkpoint, resolve_amp
from model.classifier import build_model
from utils.config import resolve_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dump", default="")
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = read_checkpoint(a.checkpoint, map_location=dev)
    cfg = ck["config"]; amp = resolve_amp(cfg)
    m = build_model(cfg).to(dev); m.load_state_dict(ck["model"]); m.eval()
    dcfg = cfg["DATASET"]
    dc = dict(resolve_data_config(m.backbone))
    if dcfg.get("crop_pct"): dc["crop_pct"] = float(dcfg["crop_pct"])
    ds = CUBDataset(resolve_path(dcfg["root"]), a.split,
                    build_transform(dc, dcfg["image_size"], train=False), dcfg["num_classes"])
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=dcfg["num_workers"])
    mains, exps, ys = [], [], []
    with torch.no_grad():
        for b in dl:
            x = b["image"].to(dev); ys.append(b["label"])
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                lg = m(x)                       # 主分支（不受 MGE 影响）
                ex = m.mge()                    # 专家 logits
            mains.append(lg.float().cpu()); exps.append(ex.float().cpu())
    M = torch.cat(mains); E = torch.cat(exps); y = torch.cat(ys)
    if a.dump:
        np.save(f"{a.dump}_main.npy", M.numpy()); np.save(f"{a.dump}_exps.npy", E.numpy())
        np.save(f"{a.dump}_labels.npy", y.numpy())
    pm = M.softmax(1); pe = E.softmax(-1)
    pred_m = pm.argmax(1); pred_e = pe.argmax(-1); k = pred_e.shape[1]
    acc = lambda p: float((p == y).float().mean() * 100)
    print(f"main      {acc(pred_m):.2f}")
    print("experts   " + "  ".join(f"{acc(pred_e[:,i]):.2f}" for i in range(k)))
    vote = torch.mode(pred_e, dim=1).values
    agree = (pred_e == vote[:, None]).sum(1)
    print(f"专家多数票 {acc(vote):.2f}（全一致 {(agree==k).float().mean()*100:.1f}%）")
    confm = pm.max(1).values
    for need in range(max(2, k - 1), k + 1):
        for tau in (0.0, 0.4, 0.5):
            use = (agree >= need) & (vote != pred_m) & (confm < tau if tau > 0 else torch.ones_like(confm, dtype=torch.bool))
            pred = pred_m.clone(); pred[use] = vote[use]
            ok = (pred == y); ok0 = (pred_m == y)
            print(f"≥{need}/{k} 一致" + (f" + main<{tau:.1f}" if tau else "") +
                  f": {float(ok.float().mean()*100):.2f} (触发 {int(use.sum())}: "
                  f"对 {int((ok&~ok0).sum())} 错 {int((~ok&ok0).sum())})")


if __name__ == "__main__":
    main()
