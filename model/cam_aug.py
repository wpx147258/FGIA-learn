"""CAM 引导的擦除 / 裁剪分支。

动机来自 analyse.ipynb 的诊断：模型**定位**已经很好（CAM 峰值 99.6% 落在 GT 框内），
但**注意力过度集中** —— 前 10/196 个格子装走了 59.7% 的质量。也就是说它不是"看不见鸟"，
是"只看鸟身上的一小块就下判断"。

两个分支各治一半：

  擦除 (drop)：把 CAM 最高的那块从特征图里抹掉，逼模型去别处找证据。
              在特征图上做，**不用重跑 backbone**，所以免费。

  裁剪 (crop)：把 CAM 最高的那块从原图裁出来放大，**重新过一遍 backbone**，
              逼模型只凭这一块也能认出鸟。
              代价是训练时间约 2x，所以默认关闭。

两路都用同一个类别标签 —— 每一路都必须独立认出这只鸟。

⚠️ **只在训练时用。** 评估走 Classifier.forward 的原路径，推理开销为零，
   测试口径和之前 17 次实验完全一致。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cam_map(feat: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """把某几类的分类得分拆到空间上（CAM）。

    feat    [B, C, h, w]   backbone 池化**前**的特征图
    weight  [K, C]         分类头权重，K = 类别数
    targets [B]            每个样本用哪一类的 CAM

    返回 [B, h, w]。定义见 analyse.ipynb：

        CAM_c[h, w] = Σ_j W_cj · F_jhw

    即"位置 (h,w) 对第 c 类得分的贡献"。它的空间平均值就是那个类的 logit（减 bias）。
    """
    w = weight[targets]                                  # [B, C]
    cam = torch.einsum("bc,bchw->bhw", w, feat)          # 把通道维 C 消掉
    return cam.relu()                                    # 负值表示"反对这一类"，只要正证据


def peak_mask(cam: torch.Tensor, alpha: float,
              min_range: float = 0.1) -> torch.Tensor:
    """把 CAM 里"足够亮"的部分做成 0/1 掩码。

    alpha 是阈值比例：alpha=0.6 表示取"CAM 最大值 60% 以上"的格子。
        alpha 小 → 抹得多（接近 0 会抹掉整张图）
        alpha 大 → 抹得少（接近 1 只抹峰值那一格）

    ⚠️ 为什么要先归一化：Mixup / CutMix 会把 CAM 的**底色整体抬高**
    （混进来的另一只鸟让背景格子也有正响应）。直接拿 alpha×max 当阈值时，
    抬高后的底色会一起越过阈值 —— 实测 CutMix 图的外接框从 16.6% 涨到 33.5%，
    局部分支就退化成"再看一遍大半张图"了。
    先减最小值再除以动态范围，等于只保留"相对这张图自己有多亮"，把底色压回 0。

    min_range 是下限保护：动态范围（max-min）小于它的图，归一化会放大噪声。
    这类图（CAM 本来就很平）退回原来的 alpha×max 判定。

    返回 [B, 1, h, w]。
    """
    hi = cam.amax(dim=(1, 2), keepdim=True)              # [B,1,1]
    lo = cam.amin(dim=(1, 2), keepdim=True)
    span = hi - lo

    normalised = (cam - lo) / span.clamp(min=1e-6)       # 拉伸到 0~1
    mask_norm = normalised >= alpha
    mask_raw = cam >= hi * alpha

    # 动态范围太小的样本（CAM 几乎是平的）用原判定，别放大噪声
    use_norm = (span >= min_range)                        # [B,1,1] bool
    mask = torch.where(use_norm, mask_norm, mask_raw)
    return mask.unsqueeze(1).float()


def mask_to_box(mask: torch.Tensor) -> torch.Tensor:
    """[B,1,h,w] 的 0/1 掩码 -> [B,4] 归一化框 (x0,y0,x1,y1)，范围 0~1。

    就是取掩码里"有 1 的地方"的外接矩形。
    """
    B, _, h, w = mask.shape
    m = mask[:, 0]                                       # [B,h,w]
    rows = torch.arange(h, device=mask.device, dtype=torch.float).view(1, h, 1)
    cols = torch.arange(w, device=mask.device, dtype=torch.float).view(1, 1, w)
    BIG = 1e4
    y0 = torch.where(m > 0, rows, torch.full_like(rows, BIG)).amin(dim=(1, 2))
    y1 = torch.where(m > 0, rows, torch.full_like(rows, -BIG)).amax(dim=(1, 2))
    x0 = torch.where(m > 0, cols, torch.full_like(cols, BIG)).amin(dim=(1, 2))
    x1 = torch.where(m > 0, cols, torch.full_like(cols, -BIG)).amax(dim=(1, 2))
    # 格子 i 覆盖像素 [i, i+1)，所以右/下边界要 +1
    box = torch.stack([x0, y0, x1 + 1, y1 + 1], dim=1)
    scale = torch.tensor([w, h, w, h], device=mask.device, dtype=torch.float)
    box = box / scale
    # 整张掩码都是 0 时上面的 min/max 会给出 [1e4, 1e4, -1e4+1, -1e4+1]，
    # 下游裁剪会拿到空切片直接崩（cbp 的 CAM 是 sketch 估出来的，会出现这种情况）。
    # 这种样本退回整图，等价于这一路不起作用。
    empty = (m > 0).flatten(1).sum(dim=1) == 0
    if empty.any():
        full = torch.tensor([0.0, 0.0, 1.0, 1.0], device=mask.device)
        box = torch.where(empty[:, None], full.expand_as(box), box)
    return box


def drop_pool(feat: torch.Tensor, mask: torch.Tensor,
              strength: float = 1.0) -> torch.Tensor:
    """削弱 mask 覆盖区域之后的池化 -> [B, C]。

    strength: 1.0 = 硬擦除（置零），0.7 = 峰值区只剩 30% 信号。

    ⚠️ 掩码只覆盖 7% 的格子（alpha=0.3），所以 strength 不能太小：
       0.3 时特征总量只衰减 2.1%，辅助分支退化成主分支的复制品。
       实测（冻结模型，alpha=0.3，主分支 86.72%，擦除后 top1）：
           strength 0.3 / 0.5 / 0.7 / 1.0  ->  86.3 / 85.2 / 81.6 / 66.8

    分母用 keep 的和而不是固定的 h*w：这样结果和 GAP 同量级，
    不需要额外的归一化层去吸收尺度差。
    """
    keep = 1.0 - strength * mask
    denom = keep.sum(dim=(2, 3)).clamp(min=1.0)          # [B,1]
    return (feat * keep).sum(dim=(2, 3)) / denom


def crop_and_resize(images: torch.Tensor, boxes: torch.Tensor,
                    out_size: int, margin: float = 1.0) -> torch.Tensor:
    """按归一化框裁剪一批图，再缩放到 out_size。

    images [B,3,H,W]   boxes [B,4] 归一化 xyxy   margin: 框的外扩倍数

    margin 是随机增强的空间 —— 它是"框往外多留多少"，比如 1.5 表示框扩大 1.5 倍面积。
    这里为了简单先做成固定倍数，要随机化就在调用处传不同的 margin。
    """
    B, _, H, W = images.shape
    patches = []
    for i in range(B):
        x0, y0, x1, y1 = boxes[i].tolist()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half_w, half_h = (x1 - x0) * margin / 2, (y1 - y0) * margin / 2
        px0 = int(max(cx - half_w, 0.0) * W)
        py0 = int(max(cy - half_h, 0.0) * H)
        px1 = min(int((cx + half_w) * W), W)
        py1 = min(int((cy + half_h) * H), H)
        # 先把左上夹进图里，再保证右下至少比左上多一格 —— 顺序反过来的话，
        # 框整个落在图外时 px0 会大于 W，max(px1, px0+1) 仍然给出空切片。
        px0, py0 = min(px0, W - 1), min(py0, H - 1)
        px1 = min(max(px1, px0 + 1), W)
        py1 = min(max(py1, py0 + 1), H)
        patch = images[i:i + 1, :, py0:py1, px0:px1]
        patches.append(F.interpolate(patch, size=(out_size, out_size),
                                     mode="bilinear", align_corners=False))
    return torch.cat(patches, dim=0)


def cam_centroid(cam: torch.Tensor) -> torch.Tensor:
    """[B,h,w] -> [B,2] 归一化质心 (x, y)，范围 0~1。

    用质心而不是峰值格：峰值格是离散的、容易落在鸟身的某一小块上；
    质心代表 CAM 质量的中心，切在它上面能保证峰值区被切成四份。
    """
    B, h, w = cam.shape
    p = cam.flatten(1)
    p = p / p.sum(1, keepdim=True).clamp(min=1e-6)
    xs = ((torch.arange(w, device=cam.device, dtype=cam.dtype) + 0.5) / w).repeat(h)
    ys = ((torch.arange(h, device=cam.device, dtype=cam.dtype) + 0.5) / h).repeat_interleave(w)
    return torch.stack([(p * xs).sum(1), (p * ys).sum(1)], dim=1)


def _reassemble(img, cx, cy, op):
    """按 (cx,cy) 切成 2×2，再按 op 重排。每块保持原始尺寸，用 cat 拼接。

    关键：不是"塞进固定槽位"。每块的宽高都不变，拼完整体仍是 H×W，
    所以内容零变形、零插值 —— 唯一变化的是四块之间的相对位置。
    """
    tl, tr = img[:, :cy, :cx], img[:, :cy, cx:]
    bl, br = img[:, cy:, :cx], img[:, cy:, cx:]
    if op == 0:   rows = [[tl, tr], [bl, br]]     # 原样（只用于验证，训练时不会取到）
    elif op == 1: rows = [[tr, tl], [br, bl]]     # 两行都左右翻
    elif op == 2: rows = [[bl, br], [tl, tr]]     # 上下对调
    elif op == 3: rows = [[br, bl], [tr, tl]]     # 上下 + 左右
    elif op == 4: rows = [[tr, tl], [bl, br]]     # 只上排翻
    else:         rows = [[tl, tr], [br, bl]]     # 只下排翻
    return torch.cat([torch.cat(r, dim=2) for r in rows], dim=1)


def quad_shuffle(images: torch.Tensor, centers: torch.Tensor,
                 min_frac: float = 0.1) -> torch.Tensor:
    """按 CAM 质心把每张图切成 2×2，随机打乱四块的位置。

    质心落在交叉点上 → 峰值区被切成四份散到四个角，"只看一处"这条路被掐断；
    其余内容全部保留，只是彼此换了位置。
    """
    B, _, H, W = images.shape
    ops = torch.randint(1, 6, (B,))
    out = []
    for i in range(B):
        cx = int(round(centers[i, 0].item() * W))
        cy = int(round(centers[i, 1].item() * H))
        # 防止切出过窄的条（太窄的条拼回去几乎看不出变化）
        cx = min(max(cx, int(min_frac * W)), int((1 - min_frac) * W))
        cy = min(max(cy, int(min_frac * H)), int((1 - min_frac) * H))
        out.append(_reassemble(images[i], cx, cy, int(ops[i])))
    return torch.stack(out)


class CamAux:
    """按 config 组装这几个分支。

    不是 nn.Module —— 它没有自己的参数，只是一段调度逻辑，所以做成普通类更清楚。
    """

    def __init__(self, drop_w: float = 0.0, crop_w: float = 0.0,
                 quad_w: float = 0.0,
                 alpha: float = 0.6, margin: float = 1.5, out_size: int = 448,
                 alpha_drop: float | None = None, alpha_crop: float | None = None,
                 drop_strength: float = 1.0, crop_union: bool = False,
                 margin_union: float | None = None, crop_majority: bool = False,
                 quad_p: float = 1.0, margin2: float | None = None,
                 teacher: str = "cam"):
        self.drop_w = float(drop_w)
        self.crop_w = float(crop_w)
        self.quad_w = float(quad_w)
        # 两个分支对"区域多大"的需求是相反的，所以阈值可以分开给：
        #   drop 要擦掉足够多的判别证据 -> alpha 小（0.2~0.3），否则擦不动
        #   crop 要框住一个能独立分类的局部 -> alpha 中等，太小会裁出半张图
        # 不单独指定时都回落到 alpha，行为与以前完全一致。
        self.alpha_drop = float(alpha if alpha_drop is None else alpha_drop)
        # 1.0 = 硬擦除；<1 时峰值区只被削弱而不是抹掉（见 drop_pool 的说明）
        self.drop_strength = float(drop_strength)
        self.alpha_crop = float(alpha if alpha_crop is None else alpha_crop)
        self.margin = float(margin)
        self.out_size = int(out_size)
        # union 裁剪：混合图上把 B 类的 CAM 也算出来，取两个 mask 的并集再框。
        # 动机（实测 48 个混合样本）：A 的框 12.0%、B 的框 11.7%，但两者 IoU 只有
        # 0.062（81% 的样本 <0.1）—— 只裁 A 的框，patch 里根本没有 B 的证据，
        # 却要求它输出 "0.6A+0.4B" 的软目标，监督是矛盾的。取并集后 patch 同时
        # 含两个区域，软目标才自洽。代价是框面积 12% → 31%（2.6 倍）。
        self.crop_union = bool(crop_union)
        # 用混合图里权重更大的那一类来算 CAM（见 __call__ 里的说明）
        self.crop_majority = bool(crop_majority)
        # quad 分支的**逐 batch 触发概率**。quad 是辅助分支，不是图像混合，
        # 想让它在「15% 的 batch」上生效就设 0.15 / 可用 batch 比例。
        self.quad_p = float(quad_p)
        # union 框和干净样本的框尺度不一样：干净图只有一个对象（框 12%），
        # 混和图要盖住两只鸟（union 31.5%）。用同一个 margin 会让两边差得更远，
        # 所以允许分开给。留空则跟随 margin。
        self.margin_union = float(margin if margin_union is None else margin_union)
        # 多视图一致性：同一个 CAM 框、用**另一个外扩倍数**再裁一张（默认 1.6），
        # 只用于和主分支做一致性，不直接接标签监督。None = 关闭。
        self.margin2 = None if margin2 is None else float(margin2)
        # 一致性教师视图的来源：cam = CAM 框裁出来的局部（默认）；
        # gt_head = 数据集按关键点裁好的头部图（来自 batch["head"]）。
        # 换成 GT 是干净的对照：一致性机制不变，只换教师视图怎么来。
        self.teacher = str(teacher)
        self.enabled = self.drop_w > 0 or self.crop_w > 0 or self.quad_w > 0

    def _cam(self, model, x, feat, labels):
        """给定类别索引 [B]，返回该类在特征图上的 CAM [B,h,w]。"""
        pos = getattr(model.head, "_pos", None)
        if pos is not None:
            w = model.head.fc.weight.detach()[labels]            # [B, dim]
            cam = torch.einsum("bd,bdp->bp", w, pos)
            # 先减掉每张图自己的均值再 relu：sketch 是带随机符号的估计，
            # 整张图的偏移可能让所有位置都是负的 —— 那样 relu 之后掩码全零，
            # crop 分支会拿到空框（v2 就是这么崩的）。减均值只去掉常数，
            # 不改变位置之间的相对高低。
            cam = (cam - cam.mean(dim=1, keepdim=True)).relu()
            return cam.view(x.shape[0], feat.shape[2], feat.shape[3])
        # 头的分类权重必须和特征图的通道维对齐。线性头直接可用；gapstd 这种
        # 「均值+标准差」拼接的头只有前一半对应空间特征，由 cam_weight() 负责切。
        get_w = getattr(model.head, "cam_weight", None)
        if get_w is None:
            raise TypeError(
                f"{type(model.head).__name__} 没有 cam_weight()，"
                f"不能和 CAM 辅助分支一起用")
        return cam_map(feat, get_w(feat.shape[1]).detach(), labels)

    def __call__(self, model, x, feat, logits, y, batch: dict | None = None):
        """跑辅助分支，返回一个 dict 交给 criterion。

        ⚠️ 掩码要 detach：它记录的是"模型**这一步**在看哪"的快照。
        如果对它求梯度，梯度会试图去改"模型该看哪"来降低损失，方向就乱了。
        """
        # 默认用 batch["label"]（= 混合图里第一个样本的原始类别 A）。
        # 但 A 不一定占多数：mixup 上实测 51% 的样本 A 权重 < 0.5，
        # 这时裁出来的 patch 主体是 B，监督用的软目标却偏向 B —— 又不一致了。
        # crop_majority=True 时改用权重更大那一类的 CAM。
        label_b = (batch or {}).get("label_b")
        cam_labels = y
        if self.crop_majority and label_b is not None and batch.get("soft_target") is not None:
            lam = batch["soft_target"].gather(1, y.view(-1, 1)).squeeze(1).to(y.device)
            cam_labels = torch.where(lam < 0.5, label_b.to(y.device), y)

        # CAM 只算一次；mask 各自算（多一次逐元素比较，可以忽略）
        cam = self._cam(model, x, feat, cam_labels)

        out = {"logits": logits}
        if self.drop_w > 0:
            mask_d = peak_mask(cam, self.alpha_drop).detach()
            out["logits_drop"] = model.head(
                drop_pool(feat, mask_d, self.drop_strength))
        if self.crop_w > 0:
            mask_c = peak_mask(cam, self.alpha_crop).detach()
            # union：把第二个类别的 CAM 也算进来，让 patch 同时覆盖两边
            margin = self.margin
            if self.crop_union and label_b is not None:
                cam_b = self._cam(model, x, feat, label_b.to(y.device))
                mask_c = torch.maximum(mask_c, peak_mask(cam_b, self.alpha_crop).detach())
                margin = self.margin_union      # 并集框本来就大，外扩要收一点
            if self.teacher == "gt_head" and batch is not None and "head" in batch:
                # GT 关键点裁出的头部图（dataset 已经做过 transform），直接当教师视图
                out["logits_crop"] = model(batch["head"].to(x.device))
                boxes = None
            else:
                boxes = mask_to_box(mask_c)
                # 把框面积也带出去：分类器可以用它做「框大才采纳裁剪分支」的门控
                wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0)
                out["cam_box_area"] = wh[:, 0] * wh[:, 1]
                patch = crop_and_resize(x, boxes, self.out_size, margin)
                out["logits_crop"] = model(patch)        # 第二次 backbone 前向
            if self.margin2 is not None and boxes is not None:
                # 第三个视角：同一个框、更大外扩。只进一致性损失，不接 CE。
                out["logits_crop2"] = model(
                    crop_and_resize(x, boxes, self.out_size, self.margin2))
        if self.quad_w > 0 and (self.quad_p >= 1.0 or torch.rand(1).item() < self.quad_p):
            # 切点用 CAM 质心：峰值区被切成四份散开，其余内容只换位置
            centers = cam_centroid(cam).detach()
            out["logits_quad"] = model(quad_shuffle(x, centers))   # 另一次 backbone 前向
        return out


def build_cam_aux(tcfg: dict, image_size: int) -> CamAux | None:
    """从 TRAIN 配置里建 CamAux；两个权重都是 0 时返回 None。"""
    aux = CamAux(
        drop_w=tcfg.get("cam_drop", 0.0),
        crop_w=tcfg.get("cam_crop", 0.0),
        quad_w=tcfg.get("cam_quad", 0.0),
        alpha=tcfg.get("cam_alpha", 0.6),
        # 这两项留空(None)时跟随 cam_alpha
        alpha_drop=tcfg.get("cam_alpha_drop"),
        alpha_crop=tcfg.get("cam_alpha_crop"),
        drop_strength=tcfg.get("cam_drop_strength", 1.0),
        margin=tcfg.get("cam_margin", 1.5),
        out_size=image_size,
        crop_union=tcfg.get("cam_crop_union", False),
        margin_union=tcfg.get("cam_margin_union"),
        crop_majority=tcfg.get("cam_crop_majority", False),
        quad_p=tcfg.get("cam_quad_p", 1.0),
        margin2=tcfg.get("cam_margin2"),
        teacher=tcfg.get("cam_teacher", "cam"),
    )
    return aux if aux.enabled else None
