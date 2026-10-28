"""诊断门 D0：模型到底缺不缺"对强扰动的稳健性"（不训练，只推理）。

候选 A（弱-强一致性）的前提是：模型在强扰动下会坏，而且坏在难类上。
这个前提从没被测过。先量，再决定要不要花三次前向去训一个一致性损失。

做法：同一个 baseline checkpoint，在 test 上跑三种视图
  clean   ：现有评测变换（短边缩放 + 中心裁剪）
  geom    ：clean 之前追加 RandAugment 几何算子（旋转/剪切/平移）
  erase   ：geom 之后再加随机擦除
判据（预注册）：clean − geom ≥ 2.0 点，或 clean-right→geom-wrong 的净损失
                ≥ 2 倍四视图 TTA 增益（约 0.7 点），否则候选 A 判 DROP。
"""
import sys, argparse
sys.path.insert(0, '.')
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from timm.data.auto_augment import rand_augment_transform
import timm

from dataset.cub import CUBDataset
from dataset.transforms import build_transform, resolve_data_config, RANDAUG_GEOMETRIC_OPS
from engine.trainer import read_checkpoint, resolve_amp
from model.classifier import build_model
from utils.config import resolve_path

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
a = p.parse_args()

dev = torch.device('cuda')
ck = read_checkpoint(a.checkpoint, map_location=dev)
cfg = ck['config']
m = build_model(cfg).to(dev); m.load_state_dict(ck['model']); m.eval()
amp = resolve_amp(cfg)
dc = cfg['DATASET']; root = resolve_path(dc['root']); size = dc['image_size']
data_cfg = resolve_data_config(m.backbone)
mean, std = data_cfg['mean'], data_cfg['std']
crop_pct = data_cfg.get('crop_pct') or 1.0
resize = int(round(size / crop_pct))
interp = timm.data.str_to_interp_mode(data_cfg.get('interpolation', 'bicubic'))

base = build_transform(data_cfg, size, train=False)
geom_aug = rand_augment_transform('rand-m7-n2-mstd0.5-inc1', {},
                                  transforms=RANDAUG_GEOMETRIC_OPS)
geom = transforms.Compose([
    transforms.Resize(resize, interpolation=interp),
    transforms.CenterCrop(size),
    geom_aug,
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
erase = transforms.Compose(list(geom.transforms) + [transforms.RandomErasing(p=1.0, value='random')])

def run(tf, tag):
    ds = CUBDataset(data_root=root, split='test', transform=tf, num_classes=dc['num_classes'])
    ld = DataLoader(ds, batch_size=cfg['EVAL']['batch_size'], shuffle=False, num_workers=6)
    correct = []
    with torch.no_grad():
        for b in ld:
            x = b['image'].to(dev)
            with torch.autocast(device_type='cuda', dtype=amp):
                lo = m(x)
            correct.append(lo.float().argmax(1).cpu() == b['label'])
    c = torch.cat(correct)
    print(f'  {tag:6s} top1 = {c.float().mean()*100:.2f}%')
    return c

print('视图对比（baseline checkpoint，test 5794）：')
cc = run(base, 'clean')
cg = run(geom, 'geom')
ce = run(erase, 'erase')
n = len(cc)
print(f'\nclean→geom : 由对转错 {int((cc & ~cg).sum())}，由错转对 {int((~cc & cg).sum())}，'
      f'净损失 {float(cc.float().mean()-cg.float().mean())*100:+.2f} 点')
print(f'clean→erase: 由对转错 {int((cc & ~ce).sum())}，由错转对 {int((~cc & ce).sum())}，'
      f'净损失 {float(cc.float().mean()-ce.float().mean())*100:+.2f} 点')
gate = float(cc.float().mean() - cg.float().mean()) * 100
print(f'\n判据：clean−geom ≥ 2.0 点 → {"通过" if gate >= 2.0 else "不通过（候选 A 应 DROP）"}'
      f'（实测 {gate:+.2f}）')
