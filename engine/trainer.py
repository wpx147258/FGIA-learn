"""训练与评估循环。
多任务（属性 / 部位监督）的扩展点只有一个：forward_batch。
"""

from __future__ import annotations

import contextlib
from datetime import datetime
from pathlib import Path

import torch

from dataset.strong_view import make_strong_view
from engine.metrics import AverageMeter, PerClassAccuracy, topk_accuracy

def resolve_amp(cfg: dict) -> torch.dtype | None:
    """把 config 里的 amp 设置转成 torch dtype；None 表示关掉混合精度。"""
    name = cfg["TRAIN"].get("amp")
    if name in (None, "none", False):
        return None
    if name != "bf16":
        raise KeyError(
            "当前只实现了 bf16（3090 原生支持，且不需要 GradScaler）。"
            f"fp16 要额外配 GradScaler，这里没写。收到: {name!r}"
        )
    if not torch.cuda.is_available():
        return None
    return torch.bfloat16


def move_batch_to_device(batch: dict, device) -> dict:
    """把 batch 里的张量搬到 device, 非张量原样保留。"""
    return {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def forward_batch(model, batch, criterion, device, amp_dtype=None):
    """搬设备 -> 前向 -> 算损失。训练和评估共用。-> (loss, outputs, logs),  
    criterion 收整个 batch dict, 多监督 target 天然拿得到
    """
    # batch = move_batch_to_device(batch, device)
    ctx = (
        contextlib.nullcontext()
        if amp_dtype is None
        else torch.autocast(device_type="cuda", dtype=amp_dtype)
    )
    with ctx:
        # 把 label 一起传进去：开了 CAM 辅助分支时模型需要它（决定算哪一类的 CAM）。
        # 评估时 model.training=False，模型会忽略 y、直接返回 logits。
        # 第三个参数把整个 batch 传进去：三流融合需要 batch["head"] / batch["body"]。
        # 不融合的模型会忽略它。
        outputs = model(batch["image"], batch.get("label"), batch)
        loss, logs = criterion(outputs, batch)
    # 开了辅助分支时 outputs 是 dict，指标只用主分支的 logits
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
    return loss, logits, logs


class RunLogger:
    """同时写 stdout / log.txt / metrics.csv。"""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "log.txt"
        self.csv_path = self.run_dir / "metrics.csv"
        self._header: list[str] | None = None
        # 续训时 metrics.csv 已存在：读回表头，否则下面的 "w" 模式会把它清空
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            with open(self.csv_path) as f:
                head = f.readline().strip()
            self._header = head.split(",") if head else None

    def log(self, msg: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def log_epoch(self, epoch: int, metrics: dict) -> None:
        """记录一轮。非标量（比如 per_class_acc 张量）会自动跳过，只进日志不进 csv。"""
        row = {"epoch": epoch}
        row.update({
            k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        })

        if self._header is None:
            self._header = list(row)
            with open(self.csv_path, "w") as f:
                f.write(",".join(self._header) + "\n")

        # 续训时当前 row 的键可能和已有表头不完全一致（比如换了辅助分支），
        # 按表头的键取值，缺的补 nan 而不是崩掉
        with open(self.csv_path, "a") as f:
            f.write(",".join(f"{row.get(k, float('nan')):.6g}" for k in self._header) + "\n")

        self.log("  ".join(f"{k}={row[k]:.4f}" for k in self._header))


def save_checkpoint(path, model, optimizer, scheduler, epoch: int,
                    best_metric: float, cfg: dict) -> None:
    """存 checkpoint。带上 optimizer/scheduler 状态是为了能断点续训。"""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_metric": best_metric,
            "config": cfg,
        },
        path,
    )


def read_checkpoint(path, map_location="cpu") -> dict:
    """只读不加载，用来先取 config（test.py 需要先知道配置才能建模型）。"""
    return torch.load(path, map_location=map_location, weights_only=True)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu") -> dict:
    """加载权重（可选带 optimizer/scheduler），返回原始 checkpoint dict。"""
    ckpt = read_checkpoint(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


class WeightAverager:
    """把训练末段若干轮的权重做等权平均（SWA）。

    存在的理由不是「涨点」，是「消掉 checkpoint 选择的运气」：
    val 只有 599 张、噪声约 1.5%，而从 50~100 个 epoch 里挑 val 最高的那个，
    选中的往往是尖峰而不是最优模型 —— 实测有一次 best.pth 比 last.pth 低 1.80 个点。

    cosine 退火到 lr≈0 时，末段各轮的权重非常接近，把它们平均掉即可消掉单轮抖动。

    只平均浮点参数。BN 的 running stats 是**缓冲区**，平均它们没有意义
    （它记录的是「某一份权重下激活的分布」），必须靠 update_bn_stats
    用平均后的权重重跑一遍数据来重算。
    """

    def __init__(self, model, start_epoch: int):
        self.start = start_epoch
        self.n = 0
        self.avg = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    def update(self, model) -> None:
        sd = model.state_dict()
        self.n += 1
        for k, v in self.avg.items():
            # avg += (new - avg) / n  等价于等权平均，但不用把所有历史都存下来
            v.add_(sd[k].detach().float().sub(v), alpha=1.0 / self.n)

    def copy_to(self, model) -> None:
        """把平均后的权重写回模型（原地，不返回新对象）。"""
        sd = model.state_dict()
        for k, v in self.avg.items():
            sd[k].copy_(v)


@torch.no_grad()
def update_bn_stats(model, loader, device, amp_dtype=None, chunk: int = 8) -> None:
    """用当前权重在训练集上只跑前向，重算 BN 的 running stats。

    SWA 之后必做：平均出来的权重对应一套**新的**激活分布，而 running stats
    还停留在最后一轮那份权重上，直接评估会有偏差（表现为指标莫名变差）。

    momentum=None 让 PyTorch 用**累积平均**而不是指数滑动平均，
    这样每张图权重相同，和 SWA 的等权平均语义一致。
    """
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None

    was_training = model.training
    model.train()
    ctx = (contextlib.nullcontext() if amp_dtype is None
           else torch.autocast(device_type="cuda", dtype=amp_dtype))
    with ctx:
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            n = batch["image"].size(0)
            # 分块跑：三流模型一次要吃 3 张 448 的图，batch=32 时会爆显存。
            # 同时把整个 batch 传进去 —— 三流融合的 forward 需要 batch["head"/"body"]，
            # 只传 image 会得到 2048 维特征，head 期望 6144 维，直接报错。
            for i in range(0, n, chunk):
                sub = {k: (v[i:i + chunk] if isinstance(v, torch.Tensor) else v)
                       for k, v in batch.items()}
                model(sub["image"], None, sub)
    model.train(was_training)


def _backbone_scale_at(cfg: dict, epoch: int, epochs: int) -> float:
    """当前轮数下 backbone 的 lr 倍率（渐进解冻）。

    配置：
        TRAIN.backbone_lr_scale        起始倍率 s0
        TRAIN.backbone_lr_scale_final  最终倍率 s1（null = 不启用，全程 s0）
        TRAIN.backbone_lr_ramp         [a, b]，占总轮数的比例，在 a~b 之间线性过渡

    动机：头是随机初始化的，前几轮梯度大、方向乱，此时给 backbone 大 lr 会被推平
    —— 实测 SWSL 权重在 scale=1.0 下 loss 卡在 ln(200)、特征 rms 从 0.65 掉到 0.17。
    等头收敛之后再把 backbone 放开，让它适配 448 分辨率和本任务。
    """
    tcfg = cfg["TRAIN"]
    s0 = float(tcfg.get("backbone_lr_scale", 1.0))
    s1 = tcfg.get("backbone_lr_scale_final")
    if s1 is None:
        return s0
    s1 = float(s1)
    a, b = tcfg.get("backbone_lr_ramp", [0.3, 0.6])
    e0, e1 = float(a) * epochs, float(b) * epochs
    if e1 <= e0:
        return s1 if epoch >= e0 else s0
    if epoch <= e0:
        return s0
    if epoch >= e1:
        return s1
    return s0 + (s1 - s0) * (epoch - e0) / (e1 - e0)


def fit(model, loaders: dict, optimizer, scheduler, criterion, device,
        cfg: dict, run_dir, logger: RunLogger | None = None,
        start_epoch: int = 0, best_top1: float = -1.0, teacher=None) -> dict:
    """主训练循环。best 按 val top1 选。

    start_epoch / best_top1 用于续训：调用方先从 checkpoint 恢复
    model / optimizer / scheduler，再把存下来的 epoch+1 和 best_metric 传进来。
    SWA 的窗口是按**绝对 epoch** 算的，所以续训不会改变窗口位置；
    但中断点之前那几轮的权重没有累积进来，续训得到的 swa.pth 会比一次跑完的粗糙。

    ⚠️ 副作用：若开启 SWA，结束时会把**平均后的权重**写回 model，
    并把 `run_dir/swa.pth` 存成这份权重。loop 内的 last.pth / best.pth 已先行落盘，
    不受影响。
    """
    logger = logger or RunLogger(run_dir)
    amp_dtype = resolve_amp(cfg)
    epochs = cfg["TRAIN"]["epochs"]
    num_classes = cfg["DATASET"]["num_classes"]

    if amp_dtype is not None:
        logger.log(f"混合精度: {amp_dtype}")

    # swa_start_frac 可以是单个数字，也可以是列表 —— 后者一次训练就产出多份
    # 不同窗口的平均权重。改窗口本身不影响训练轨迹（它只在训练结束后起作用），
    # 所以窗口扫描没必要各跑一遍完整的 22 分钟。
    swas: list[tuple[float, WeightAverager]] = []
    if cfg["TRAIN"].get("swa", True):
        frac = cfg["TRAIN"].get("swa_start_frac", 0.7)
        fracs = list(frac) if isinstance(frac, (list, tuple)) else [frac]
        swas = [(f, WeightAverager(model, start_epoch=int(epochs * f))) for f in fracs]
        logger.log(f"SWA: 将各从 epoch {[s.start for _, s in swas]} 起累积平均权重"
                   f"（共 {epochs} 轮）")

    for epoch in range(start_epoch, epochs):
        # P×K 采样器按 epoch 重新洗牌（成对判别用）。普通 shuffle 没有这个方法，
        # getattr + hasattr 一起挡住。
        bs = getattr(loaders["train"], "batch_sampler", None)
        if hasattr(bs, "set_epoch"):
            bs.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device, epoch,
            amp_dtype=amp_dtype,
            grad_clip=cfg["TRAIN"]["grad_clip"],
            log_every=cfg["TRAIN"]["log_every"],
            logger=logger,
            strong_view=bool(cfg["TRAIN"].get("strong_view", False)),
            strong_erase_p=float(cfg["TRAIN"].get("strong_erase_p", 0.0)),
            teacher=teacher,
        )
        val_metrics = evaluate(
            model, loaders["val"], criterion, device,
            amp_dtype=amp_dtype, num_classes=num_classes,
        )

        # 调度器在 optimizer.step() 之后、按 epoch 推进
        if scheduler is not None:
            scheduler.step()

        # 渐进解冻：scheduler.step() 把每组 lr 重置成 base_lr × lambda(epoch)，
        # 这里再给 backbone 组乘一个「当前 scale / 起始 scale」，等价于让
        # backbone_lr_scale 随轮数从 s0 线性升到 s1（见 _backbone_scale_at）。
        bb_scale = _backbone_scale_at(cfg, epoch, epochs)
        s0 = float(cfg["TRAIN"].get("backbone_lr_scale", 1.0))
        # 分层学习率下每层倍率不同，渐进解冻的「整体乘一个比值」不适用
        if any(v is not None for v in (cfg["TRAIN"].get("backbone_lr_layers") or {}).values()):
            s0 = 0.0
        if s0 > 0:
            ratio = bb_scale / s0
            for pg in optimizer.param_groups:
                if pg.get("part") == "backbone":
                    pg["lr"] *= ratio

        # 开了判别式学习率后各组的 lr 不同，记最大的那个（= 头的 lr = config 里的 lr）
        row = {"lr": max(pg["lr"] for pg in optimizer.param_groups),
               "bb_scale": bb_scale}
        row.update({(k if k.startswith("train_") else f"train_{k}"): v
                    for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        logger.log_epoch(epoch, row)

        is_best = val_metrics["top1"] > best_top1
        if is_best:
            best_top1 = val_metrics["top1"]
            save_checkpoint(run_dir / "best.pth", model, optimizer, scheduler,
                            epoch, best_top1, cfg)
        save_checkpoint(run_dir / "last.pth", model, optimizer, scheduler,
                        epoch, best_top1, cfg)

        for _, s in swas:
            if epoch >= s.start:
                s.update(model)

    logger.log(f"训练结束。best val top1 = {best_top1:.2f}")

    if start_epoch >= epochs:
        logger.log(f"start_epoch={start_epoch} >= epochs={epochs}，没有需要训练的轮次")
        return {"best_top1": best_top1, "last_val": float("nan"), "swa_val": {}}

    result = {"best_top1": best_top1, "last_val": val_metrics["top1"], "swa_val": {}}
    if swas:
        # 每做一份 SWA 都会改写 model 的权重和 BN 缓冲，所以先留一份原件，
        # 每份之间复位，收尾时再复位回去（避免 fit 的调用方拿到被改过的 model）。
        origin = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for frac, s in swas:
            if s.n <= 1:
                logger.log(f"SWA[{frac}]: 窗口内只有 {s.n} 轮，平均没有意义，跳过")
                continue
            expected = epochs - s.start
            if s.n < expected:
                # WeightAverager 的累积状态不在 checkpoint 里，每次续训都会清零，
                # 所以中断过的 run 只能攒到续训之后的那几轮。
                logger.log(f"⚠️ SWA[{frac}]: 只累积了 {s.n}/{expected} 轮"
                           f"（续训导致窗口不完整），这份平均权重仅供参考")
            model.load_state_dict(origin)
            s.copy_to(model)
            update_bn_stats(model, loaders["train"], device, amp_dtype)
            m = evaluate(model, loaders["val"], criterion, device,
                         amp_dtype=amp_dtype, num_classes=num_classes)
            logger.log(
                f"SWA[{frac}] 平均最后 {s.n} 轮  val top1 = {m['top1']:.2f}   "
                f"(对照: best={best_top1:.2f}  last={val_metrics['top1']:.2f})"
            )
            name = "swa.pth" if len(swas) == 1 else f"swa_f{frac}.pth"
            save_checkpoint(run_dir / name, model, optimizer, scheduler,
                            epochs - 1, m["top1"], cfg)
            result["swa_val"][frac] = m["top1"]
        model.load_state_dict(origin)

    return result


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, *,
                    amp_dtype=None, grad_clip: float = 0.0, log_every: int = 50,
                    logger: RunLogger | None = None,
                    strong_view: bool = False,
                    strong_erase_p: float = 0.0,
                    teacher=None) -> dict[str, float]:
    """跑一个 epoch ->  {'loss', 'top1', 'top5'}。"""
    model.train()
    loss_meter = AverageMeter()
    acc1 = AverageMeter()
    acc5 = AverageMeter()
    # 损失里各项的分量（entropy / consistency / pair_* / 各路辅助）都要留下来，
    # 否则「开了但没生效」这种情况在日志里完全看不出来
    log_meters: dict[str, AverageMeter] = {}
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        batch = move_batch_to_device(batch, device) 
        # 强视图在**混图之后**的张量上生成，自动继承同一组 mix 标签；
        # 不去碰弱视图的训练变换（否则基座变成双变量）
        if strong_view:
            batch["image_strong"] = make_strong_view(
                batch["image"], erase_p=strong_erase_p)
        if teacher is not None:
            # 教师看的是同一张（可能已混合的）图，只取 logits，不参与反传
            with torch.no_grad():
                t_logits = teacher(batch["image"])
            batch["teacher_logits"] = (t_logits.float() if torch.is_tensor(t_logits)
                                       else t_logits["logits"].float())
        loss, outputs, sub_logs = forward_batch(model, batch, criterion, device, amp_dtype)
        batch_n = batch["image"].size(0)
        loss_meter.update(loss, batch_n)
        for k, v in sub_logs.items():
            log_meters.setdefault(k, AverageMeter()).update(v, batch_n)
        acc = topk_accuracy(outputs, batch["label"])
        acc1.update(acc["top1"], batch_n)
        acc5.update(acc["top5"], batch_n)
        
        loss.backward()
        if grad_clip > 0: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    out = {"loss": loss_meter.avg, "top1": acc1.avg, "top5": acc5.avg}
    # 不加 train_ 前缀：fit 里统一加，否则会变成 train_train_xxx
    out.update({k: m.avg for k, m in log_meters.items()})
    ap = getattr(model, "attn_pool", None)
    if ap is not None:
        out["attn_out_w"] = float(ap.out.weight.detach().norm())
        if getattr(ap, "_last_out_norm", None) is not None:
            out["attn_out_norm"] = float(ap._last_out_norm)
        if getattr(ap, "_last_ratio", None) is not None:
            out["attn_gap_ratio"] = float(ap._last_ratio)
    # GlobalContext 挂在 backbone 上，没有上面对应的日志就看不出它有没有长起来
    gc = getattr(getattr(model, "backbone", None), "global_ctx", None)
    if gc is not None:
        out["gc_out_norm"] = float(gc._last_out_norm)
        out["gc_gap_ratio"] = float(gc._last_ratio)
        out["gc_out_w"] = float(gc.out.weight.detach().norm())
    for bname in ("cam_beta", "cam_beta2", "hyp_beta1", "hyp_beta2"):
        b = getattr(model, bname, None)
        if b is not None:
            out[bname] = float(b.detach())
    # 多元素的 β（部件 logit 级融合）逐个记录
    b = getattr(model, "pa_beta", None)
    if b is not None:
        for i, v in enumerate(b.detach().flatten().tolist()):
            out[f"pa_beta{i}"] = float(v)
    pa = getattr(getattr(model, "backbone", None), "part_attn", None)
    if pa is not None:
        if getattr(pa, "out", None) is not None:
            out["pa_out_w"] = float(pa.out.weight.detach().norm())
        for attr, key in (("_last_entropy", "pa_entropy"), ("_last_overlap", "pa_overlap"),
                          ("_last_ratio", "pa_ratio")):
            v = getattr(pa, attr, None)
            if v is not None:
                out[key] = float(v)
    return out


@torch.no_grad()
def evaluate(model, loader, criterion, device, *, amp_dtype=None,
             num_classes: int = 200) -> dict:
    """在验证 / 测试集上跑一遍。

    返回（标量部分会被写进 metrics.csv）：
        {"loss": float, "top1": float, "top5": float, "macro_acc": float,
         "per_class_acc": Tensor[num_classes], "per_class_count": Tensor[num_classes]}
    """
    model.eval()
    per_class = PerClassAccuracy(num_classes)
    loss_meter, acc1, acc5 = AverageMeter(), AverageMeter(), AverageMeter()
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        loss, outputs, _ = forward_batch(model, batch, criterion, device, amp_dtype)
        
        loss_meter.update(loss)
        acc = topk_accuracy(outputs, batch["label"])
        acc1.update(acc["top1"])
        acc5.update(acc["top5"])
        per_class.update(outputs, batch["label"])
    
    pc = per_class.compute()
    return {
        "loss": loss_meter.avg, "top1": acc1.avg, "top5": acc5.avg, 
        "macro_acc": pc["macro"], "per_class_acc": pc["per_class_acc"], "per_class_count": pc["per_class_count"]
    }        