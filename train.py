"""训练入口。

用法：
    python train.py --config configs/baseline_config.yaml
    python train.py --config configs/baseline_config.yaml --set TRAIN.lr=0.0005 TRAIN.batch_size=64
冒烟测试：
    python train.py --config configs/baseline_config.yaml --set TRAIN.epochs=2
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from dataset import build_dataloaders
from engine.build import build_criterion, build_optimizer, build_scheduler
from engine.trainer import RunLogger, fit, load_checkpoint
from model.classifier import build_model
from utils.config import BASE_CONFIG, dump_config, load_config, resolve_path
from utils.seeds import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="CUB-200-2011 训练")
    parser.add_argument("--config", required=True,
                        help="实验配置，如 configs/baseline_config.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="覆盖配置项，如 --set TRAIN.lr=0.0005")
    parser.add_argument("--resume", nargs="?", const=True, default=None,
                        metavar="CKPT",
                        help="从 checkpoint 续训。不带路径时自动找 "
                             "outputs/<EXP_NAME>/ 下最新的 last.pth")
    return parser.parse_args()


def _find_latest_last(cfg: dict) -> Path | None:
    """outputs/<EXP_NAME>/ 下最新（按修改时间）的 last.pth。"""
    root = resolve_path(cfg["OUTPUT"]["dir"]) / cfg["EXP_NAME"]
    cands = sorted(root.glob("*/last.pth"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def main() -> None:
    args = parse_args()

    # 1) 配置：base <- 实验配置 <- --set
    cfg = load_config([BASE_CONFIG, args.config], args.set)

    # 1b) 续训：模型结构 / 超参必须和中断时一致，所以 checkpoint 里的配置优先
    resume_path = None
    if args.resume:
        if isinstance(args.resume, str):
            resume_path = Path(args.resume).resolve()
        else:
            resume_path = _find_latest_last(cfg)
        if resume_path is None or not resume_path.exists():
            raise SystemExit(f"[resume] 找不到 checkpoint: {args.resume}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
        cfg = ckpt["config"]
        if args.set:
            print("[resume] 警告：--set 被忽略，续训使用 checkpoint 里的配置")
        print(f"[resume] {resume_path}  epoch={ckpt['epoch']}  "
              f"best_val={ckpt['best_metric']:.2f}")

    set_seed(cfg["SEED"])

    # 2) 运行目录：新训练用新时间戳；续训写回原目录（metrics.csv 追加）
    if resume_path is not None:
        run_dir = resume_path.parent
    else:
        run_dir = (
            resolve_path(cfg["OUTPUT"]["dir"])
            / cfg["EXP_NAME"]
            / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        dump_config(cfg, run_dir / "config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = RunLogger(run_dir)
    logger.log(f"run_dir = {run_dir}")
    logger.log(f"device  = {device}")

    # 3) 模型。注意顺序：**先建模型，再建 dataloader** ——
    #    transform 的归一化参数要从 backbone 的 pretrained_cfg 解析出来。
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"model   = {cfg['MODEL']['name']}  可训练参数 {n_params / 1e6:.2f}M")

    # 4) 数据
    loaders = build_dataloaders(cfg, model.backbone)
    logger.log(f"samples num in train, val and test: {len(loaders["train"].dataset), len(loaders["val"].dataset), len(loaders["test"].dataset)}")

    # 5) 损失 / 优化器 / 调度器
    # 知识蒸馏：教师只用于训练，推理路径完全不变
    teacher = None
    _dck = cfg["TRAIN"].get("distill_ckpt") or ""
    if _dck:
        from engine.trainer import read_checkpoint
        from model.classifier import build_model as _bm
        _tck = read_checkpoint(resolve_path(_dck), map_location=device)
        teacher = _bm(_tck["config"]).to(device)
        teacher.load_state_dict(_tck["model"])
        teacher.eval()
        for _p in teacher.parameters():
            _p.requires_grad_(False)
        print(f"[distill] 教师已加载: {_dck}  权重 {cfg['TRAIN'].get('distill_w', 0.0)}  T={cfg['TRAIN'].get('distill_T', 2.0)}")

    criterion = build_criterion(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # 5b) 续训：恢复 weights / optimizer / scheduler，以及轮次和 best
    start_epoch, best_top1 = 0, -1.0
    if resume_path is not None:
        ck = load_checkpoint(resume_path, model, optimizer, scheduler,
                             map_location=device)
        start_epoch = int(ck["epoch"]) + 1
        best_top1 = float(ck["best_metric"])
        logger.log(f"续训：从中断处 epoch {start_epoch} 开始（已训 "
                   f"{start_epoch}/{cfg['TRAIN']['epochs']} 轮），"
                   f"best_val={best_top1:.2f}")
        if start_epoch >= cfg["TRAIN"]["epochs"]:
            logger.log("训练已经跑完，无需续训")
            return

    # 6) 开训
    result = fit(model, loaders, optimizer, scheduler, criterion, device, cfg,
                 run_dir, logger, start_epoch=start_epoch, best_top1=best_top1,
                 teacher=teacher)

    # 训练结束顺手画一张曲线（失败不影响主流程）
    try:
        from scripts.plot_curves import plot as plot_curves
        plot_curves(run_dir, force=True)
        logger.log(f"曲线: {run_dir}/curves.png")
    except Exception as e:                              # noqa: BLE001
        logger.log(f"[warn] 画曲线失败: {e}")
    print(result)
    logger.log(f"权重: {run_dir}/best.pth")
    logger.log(f"测试: python test.py --checkpoint {run_dir}/best.pth")

if __name__ == "__main__":
    main()
