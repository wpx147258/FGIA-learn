"""机制诊断：196 个 token 里有没有"干扰 token"，值不值得剪。

问题：我们的自注意力模块在 196 个 token 上让它们互相看，结果与"不看、只做同样的池化"打平。
一个可能的解释是：这些 token 里大部分是无信息的，甚至带反信息（背景/无关区域），
所以"让它们互相看"只是把噪声平均得更均匀。

这个脚本量三件事（纯推理，不训练）：
  1. 每个 token 对真实类 logit 的贡献 a_k = w_y · f_k，看它有多集中
  2. 贡献为负的 token 占多少（这些是"反证"，剪掉它们理论上该有帮助）
  3. 落在 GT 框外的 token 与框内的 token，贡献分布差多少

如果 a_k 高度集中且大量为负，那"按贡献选择 token"这一族方法就有明确的操作空间；
如果 a_k 分布平坦、负值稀少，那剪枝也没多少可剪的。

用法：
    python scripts/probe_tokens.py --checkpoint <ckpt> [--n 500]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.cub import CUBDataset
from dataset.transforms import build_transform, resolve_data_config
from engine.trainer import read_checkpoint, resolve_amp
from model.classifier import build_model
from torch.utils.data import DataLoader
from utils.config import resolve_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=800, help="取多少张测试图做统计")
    a = ap.parse_args()

    dev = torch.device("cuda")
    ck = read_checkpoint(a.checkpoint, map_location=dev)
    cfg = ck["config"]
    m = build_model(cfg).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    amp = resolve_amp(cfg)
    W = m.head.fc.weight.detach().float()          # [200, 2048]
    dc = cfg["DATASET"]
    ds = CUBDataset(data_root=resolve_path(dc["root"]), split="test",
                    transform=build_transform(resolve_data_config(m.backbone),
                                              dc["image_size"], train=False),
                    num_classes=dc["num_classes"])
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    # GT 框直接读文件：dataset 只有在 use_parts=True 时才加载 bboxes，
    # 这里只是诊断，不想为它多开一条数据链路
    boxes = {}
    with open(resolve_path(dc["root"]) / "bounding_boxes.txt") as f:
        for line in f:
            p_ = line.split()
            if p_:
                boxes[int(p_[0])] = tuple(map(float, p_[1:5]))

    pos_frac, neg_frac, top20_share, in_ratio = [], [], [], []
    contrib_in, contrib_out = [], []
    seen = 0
    with torch.no_grad():
        for b in loader:
            x = b["image"].to(dev)
            with torch.autocast(device_type="cuda", dtype=amp):
                feat = m.backbone.forward_features(x)      # [B,2048,14,14]
            feat = feat.float()
            B, C, H, Wd = feat.shape
            f = feat.flatten(2).transpose(1, 2)            # [B,196,2048]
            y = b["label"].to(dev)
            w = W[y]                                       # [B,2048]
            contrib = torch.einsum("bnc,bc->bn", f, w)     # 每 token 对该类 logit 的贡献
            pos_frac.append((contrib > 0).float().mean(1).cpu())
            neg_frac.append((contrib < 0).float().mean(1).cpu())
            srt = contrib.sort(dim=1, descending=True).values
            k = int(0.2 * contrib.size(1))
            pos = contrib.clamp(min=0)
            top20_share.append((srt[:, :k].clamp(min=0).sum(1)
                                / pos.sum(1).clamp(min=1e-6)).cpu())

            # 框内 / 框外 token 的贡献（用 GT 框，仅诊断）
            base = feat.mean(dim=(2, 3)) / 196             # 每个 token 的平均贡献尺度
            scale = base.norm(dim=1, keepdim=True).clamp(min=1e-6)
            for i in range(B):
                img_id = b["image_id"][i].item()
                p = ds.imgs_paths[img_id]
                Wi, Hi = Image.open(p).size
                bb = boxes.get(img_id)
                if bb is None:
                    continue
                bx, by, bw, bh = bb
                g = 14
                c0 = min(max(int(bx / Wi * g), 0), g - 1)
                c1 = min(max(int(np.ceil((bx + bw) / Wi * g)), c0 + 1), g)
                r0 = min(max(int(by / Hi * g), 0), g - 1)
                r1 = min(max(int(np.ceil((by + bh) / Hi * g)), r0 + 1), g)
                mask = torch.zeros(g, g, dtype=torch.bool)
                mask[r0:r1, c0:c1] = True
                mask = mask.flatten().to(contrib.device)
                if mask.any():
                    contrib_in.append(contrib[i][mask].mean().cpu())
                if (~mask).any():
                    contrib_out.append(contrib[i][~mask].mean().cpu())
            seen += B
            if seen >= a.n:
                break

    pf = torch.cat(pos_frac).mean().item() * 100
    nf = torch.cat(neg_frac).mean().item() * 100
    ts = torch.cat(top20_share).mean().item() * 100
    ci = torch.stack(contrib_in).mean().item()
    co = torch.stack(contrib_out).mean().item()
    print(f"样本 {seen} 张，每张 196 个 token\n")
    print(f"贡献为正的 token 占比       {pf:5.1f}%")
    print(f"贡献为负的 token 占比       {nf:5.1f}%   （这些是「反证」，剪掉理论上有利）")
    print(f"贡献最高的 20% token 占总额  {ts:5.1f}%   （50% 说明很集中，20% 说明很平均）")
    print(f"\n框内 token 的平均贡献  {ci:+.4f}")
    print(f"框外 token 的平均贡献  {co:+.4f}")
    print(f"框外 / 框内 = {co / ci if ci != 0 else float('nan'):+.2f}")

    # 推理期的 token 选择：用**模型自己的预测类**算贡献，取 top-k token 平均后重分类。
    # 这里不算标签泄漏（没有用真值），可以当作"若能挑出有用 token，上限在哪"的参照。
    acc = {r: [0, 0] for r in (1.0, 0.5, 0.25, 0.1)}
    with torch.no_grad():
        seen = 0
        for b in loader:
            x = b["image"].to(dev)
            with torch.autocast(device_type="cuda", dtype=amp):
                feat = m.backbone.forward_features(x)
            feat = feat.float()
            B, C, H, Wd = feat.shape
            fmap = feat
            f = fmap.flatten(2).transpose(1, 2)
            pooled = fmap.mean(dim=(2, 3))
            pred = m.head(pooled).argmax(1)
            wp = W[pred]
            c = torch.einsum("bnc,bc->bn", f, wp)
            for r in acc:
                k = max(1, int(r * f.size(1)))
                idx = c.topk(k, dim=1).indices
                sel = f.gather(1, idx.unsqueeze(-1).expand(-1, -1, C)).mean(1)
                p2 = m.head(sel).argmax(1)
                acc[r][0] += int((p2 == b["label"].to(dev)).sum())
                acc[r][1] += B
            seen += B
            if seen >= a.n:
                break
    print("\n按预测类贡献选 top-k token 后重新池化的准确率（与 GAP 对比）：")
    for r, (ok, n) in acc.items():
        print(f"  保留 {int(r*100):3d}% 的 token  {ok/n*100:6.2f}%")


if __name__ == "__main__":
    main()
