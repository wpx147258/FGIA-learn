from __future__ import annotations

import torch
import torch.nn as nn

from model.attention import build_attention_aux
from model.backbone import build_backbone
from model.cam_aug import build_cam_aux, cam_map, crop_and_resize, mask_to_box, peak_mask
from model.fpn import build_fpn
from model.global_ctx import build_global_context
from model.mge import build_mge
from model.regions import RegionRouter, propose_regions, routing_features
from model.part_attn import build_part_attention
from model.head import HEADS
from model.pair import PairAux
from model.attn_pool import AttnPool, SEGate, TokenLearner
from model.pmg import PMGHead, jigsaw


class Classifier(nn.Module):
    """把 backbone 和 head 串起来。

    可选挂一个 CAM 辅助分支（见 model/cam_aug.py）—— **只在训练时生效**，
    评估路径和不挂时逐位相同。
    """

    def __init__(self, backbone: nn.Module, head: nn.Module,
                 cam_aux=None, attn_aux: nn.Module | None = None, fpn=None,
                 fusion: str = "", pair_aux: nn.Module | None = None,
                 pmg: nn.Module | None = None, deep_sup: nn.Module | None = None,
                 attn_pool: nn.Module | None = None, attn_from: str = "layer4",
                 se_gate: nn.Module | None = None,
                 tokenlearner: nn.Module | None = None,
                 cam_fuse: bool = False, cam_fuse_regions: int = 1, cam_gate_area: float = 0.0,
                 part_rerank: bool = False, pr_margin: float = 1.3,
                 pr_alpha: float = 0.2, pr_size: int = 448, pr_parts: int = 0,
                 pr_use_cam: bool = False,
                 hyp_fuse: bool = False, hyp_margin: float = 1.3,
                 hyp_alpha: float = 0.2, hyp_size: int = 448,
                 hyp_combiner: bool = False, part_train_aux: bool = False,
                 pta_parts: int = 2, pta_margin: float = 1.3, pta_alpha: float = 0.6,
                 pta_cons_w: float = 0.0, pta_cons_T: float = 2.0,
                 pta_use_cam: bool = False, pta_cam_alpha: float = 0.2,
                 pa_logit_fuse: bool = False, pa_parts: int = 4, pa_veto: bool = False,
                 pa_veto_tau: float = 0.0, freeze_backbone: bool = False,
                 mge=None, region_route: bool = False, rr_regions: int = 2,
                 rr_alpha: float = 0.2, rr_margin: float = 1.3,
                 rr_hidden: int = 64, rr_size: int = 448):
        super().__init__()
        self.backbone = backbone
        self.head = head
        # 测试时把「按 CAM 框裁剪再前向一次」的那一路并进最终预测。
        # 机理性说明：这不是 TTA（同一张图换视角平均），而是模型结构的一部分——
        # 一路整图、一路模型自己选的判别区，融合权重 β 也是训练出来的。
        # 代价是测试 2 次前向。β 初始为 0 => 初始测试与基线逐位等价。
        self.cam_fuse = bool(cam_fuse)
        self.cam_fuse_regions = int(cam_fuse_regions)
        # 按 CAM 框面积门控裁剪分支：只有框足够大（判别区本来就大）时才采纳它。
        # 依据（scripts/probe_cam_box.py，test 200 张）：框大的三分位上裁剪分支比主分支
        # 高 5.9 点，框小的三分位上低 11.9 点。0 = 不门控。
        self.cam_gate_area = float(cam_gate_area)
        # 部件裁出来重新编码（MA-CNN 的做法）+ 部件间自注意力融合。
        # 测试 K+1 次前向；这是机制本身需要，不是 TTA。
        # 假设条件重检：整图 -> 取 top-2 类别 -> 各自算 CAM 框、裁出来重编码 -> 融合。
        # 动机（零训练诊断）：整图 + top1 框的 oracle 90.01，再加上 top2 框的 oracle 91.39，
        # 且 top2 框单独救回 131 张整图错的样本——不同类别假设指向不同的证据位置。
        # 测试 3 次前向，属于机制需要，不是 TTA。
        self.hyp_fuse = bool(hyp_fuse)
        self.hyp_margin = float(hyp_margin)
        self.hyp_alpha = float(hyp_alpha)
        self.hyp_size = int(hyp_size)
        # 训练期部件辅助分支：把部件框裁出来重编码，只加辅助 CE / 一致性，
        # 测试仍然是**单次前向的整图**。动机：CAM 裁剪分支的 +1.19 本来就是
        # 训练期效应（测试不用裁剪），部件裁剪同理——但训练期没有代价。
        # 部件证据走 logit 级融合：每个部件的描述子用它自己那组通道对应的
        # 主分类器权重切片算 logits（权重绑定，零新参数），再按可学习 β_k 加回主 logits。
        # 依据：pa4 的四个部件单独就有 86–89%（oracle 93.3），但原来的出口
        # （proj→注意力→mean→Linear→残差）把它们丢掉了（head(y) 单独只有 0.83%）。
        # 一致否决（veto）：若 K 个部件**全部**同意某个类、且与主分支不同、且一致票
        # 置信度高于主分支，就把该类的 logit 抬上去。零参数、零额外前向、不改训练。
        # 离线分析（scripts/probe_part_gate.py）：pa4 上触发 26 张，改对 17 改错 5。
        self.pa_veto = bool(pa_veto)
        # τ>0 时用 Chow 式的「主分支不确定才否决」：主分支 max prob < τ 且部件全一致
        # 指向另一个类才改判。离线实测（pa4 权重）：触发子集上主分支只有 5% 对、
        # 否决 74% 对；τ 在 0.3–0.5 之间结果稳定（89.32–89.37）。
        self.pa_veto_tau = float(pa_veto_tau)
        self.freeze_backbone = bool(freeze_backbone)
        # 多粒度专家（layer2/3/4 GAP + layer4 GMP），只用于训练期辅助 CE；
        # 测试时可以当否决票（离线分析），不影响主分支输出。
        self.mge = mge
        # 多区域重编码 + 学习式路由（见 model/regions.py）。
        # 测试 1+K 次前向；初始化时 fused = main（router 输出零 + gamma 0）。
        self.region_route = bool(region_route)
        self.rr_regions = int(rr_regions)
        self.rr_alpha = float(rr_alpha)
        self.rr_margin = float(rr_margin)
        self.rr_size = int(rr_size)
        if self.region_route:
            fd = int(getattr(backbone, "num_features"))
            in_dim = 3 + self.rr_regions * 4
            self.region_router = RegionRouter(in_dim, self.rr_regions, hidden=int(rr_hidden))
        self.pa_logit_fuse = bool(pa_logit_fuse)
        if self.pa_logit_fuse:
            self.pa_beta = nn.Parameter(torch.zeros(int(pa_parts)))
        self.part_train_aux = bool(part_train_aux)
        self.pta_parts = int(pta_parts)
        self.pta_margin = float(pta_margin)
        self.pta_alpha = float(pta_alpha)
        self.pta_cons_w = float(pta_cons_w)
        self.pta_cons_T = float(pta_cons_T)
        # pta_use_cam=True：辅助裁剪改用「top-2 预测类的 CAM 框」，而不是部件掩码。
        # 依据：layer4 上部件掩码空间上散（框≈整图），只有 CAM 是紧凑的；
        # 而 hyp_fuse 又显示 β2（第二候选的区域）会被模型赋予正权重。
        self.pta_use_cam = bool(pta_use_cam)
        self.pta_cam_alpha = float(pta_cam_alpha)
        self.hyp_combiner = bool(hyp_combiner) and self.hyp_fuse
        if self.hyp_fuse:
            self.hyp_beta1 = nn.Parameter(torch.zeros(1))
            self.hyp_beta2 = nn.Parameter(torch.zeros(1))
        if self.hyp_combiner:
            # 用一个小注意力模块替代固定的 β 加权：三路 logits 作为 3 个 token 互相看，
            # 自己学「什么时候该信哪一路」。零初始化 => 初始输出等于整图。
            self.hyp_comb = nn.Module()
            self.hyp_comb.proj = nn.Linear(getattr(head, "fc").out_features, 128)
            self.hyp_comb.emb = nn.Parameter(torch.zeros(1, 3, 128))
            layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256,
                                               dropout=0.0, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.hyp_comb.blocks = nn.TransformerEncoder(layer, num_layers=1)
            self.hyp_comb.norm = nn.LayerNorm(128)
            self.hyp_comb.out = nn.Linear(128, getattr(head, "fc").out_features)
            nn.init.zeros_(self.hyp_comb.out.weight)
            nn.init.zeros_(self.hyp_comb.out.bias)
        self.part_rerank = bool(part_rerank)
        # 重新编码只用前 pr_parts 个部件（0 = 全部）；按掩码质量选，
        # 这样检测器可以是 K=4，但测试只多跑 2 次前向。
        self.pr_parts = int(pr_parts)
        # 在与部件重编码并列的位置再加一路「按 CAM 框裁剪重编码」。
        # 对照实验显示单独这一路就值 +1.19（r50_nocam 88.21 vs 基线 89.40），
        # 而部件重编码换掉它是不划算的——所以正确做法是把两者叠起来（4 次前向）。
        self.pr_use_cam = bool(pr_use_cam)
        if self.part_rerank and self.pr_use_cam:
            self.pr_cam_beta = nn.Parameter(torch.zeros(1))
        self.pr_margin = float(pr_margin)
        self.pr_alpha = float(pr_alpha)
        self.pr_size = int(pr_size)
        if self.cam_fuse:
            self.cam_beta = nn.Parameter(torch.zeros(1))
        if self.part_rerank:
            from model.part_attn import PartFusion
            fd = int(getattr(backbone, "num_features"))
            nc = int(getattr(getattr(head, "fc", None), "out_features", 200))
            self.part_fusion = PartFusion(channels=fd)
            # 部件自己的分类头：不与主分类器共享，避免部件损失直接改写主头
            self.part_head = nn.Linear(fd, nc)
        if self.cam_fuse and self.cam_fuse_regions >= 2:
            # 第二路用更大的外扩（cam_margin2），近似「身体/上下文」视角。
            # 零训练诊断：GT 头 + GT 身体 + 整图三路的 oracle 是 91.85，
            # 远高于单区域（只有 CAM 框时 oracle 90.01）。
            self.cam_beta2 = nn.Parameter(torch.zeros(1))
        # "" = 单流；"head_body" = 三流（整图 / 头部 / 身体）共享 backbone
        self.fusion = fusion
        # 多尺度融合（见 model/fpn.py）。None = 不融合，走和以前逐位相同的路径。
        # ⚠️ 它有参数但既不在 backbone 也不在 head 里，build_optimizer 必须单独收它。
        self.fpn = fpn
        # CamAux 不是 nn.Module（它没有自己的参数），挂在普通属性上
        self.cam_aux = cam_aux
        # AttentionAux 有参数（1×1 conv + 部位分类头），挂在属性上会被自动注册成子模块，
        # 这样 .cuda() / .parameters() / state_dict 都能看到它
        self.attn_aux = attn_aux
        # 成对判别（API-Net）。有参数（mutual vector 的 MLP），要注册成子模块，
        # 否则 .parameters() 收不到它、优化器不会更新它。
        self.pair_aux = pair_aux
        # PMG（见 model/pmg.py）：每 batch 跑 n=8/4/2/原图 四次前向，
        # 输出四路 logits；推理只用原图那一路（单模型，不是集成）
        self.pmg = pmg
        # 深度监督：在 layer3 上加一路辅助分类器（只训练时用，见 build_model）
        self.deep_sup = deep_sup
        # 部位间自注意力：输出与 GAP 同维并相加（零初始化，初始等价于基线）
        self.attn_pool = attn_pool
        # replace=True 时模块直接产出「替换 GAP」的池化特征，而不是残差
        self.attn_replace = bool(getattr(attn_pool, "replace", False))
        # 自注意力可以接 layer4 也可以接 layer3（28x28，784 token）
        self.attn_from = attn_from
        # 最终特征图上的通道门控（乘法，初始≈恒等）
        self.se_gate = se_gate
        # 软空间注意力池化（TokenLearner）：输出与 GAP 同维相加，零初始化
        self.tokenlearner = tokenlearner
        self.tl_replace = bool(getattr(tokenlearner, "replace", False))

    def _capture_parts(self):
        """取走部件注意力模块在**本次主干前向**里算出的部件描述子与散度项。

        必须在 CAM 裁剪分支再跑一次主干之前调用，否则拿到的是裁剪图上的部件。
        返回 (z, div)；推理时不需要（辅助损失只在训练时用），返回 None。
        """
        pa = getattr(self.backbone, "part_attn", None)
        if pa is None or not self.training or pa._last_part_logits is None:
            return None
        return (pa._last_part_logits, pa._last_div)

    def _forward_region_route(self, x: torch.Tensor):
        """整图 + K 个 CAM 区域重编码，逐样本路由融合。"""
        feat = self.backbone.forward_features(x)
        pooled = self.backbone.forward_head(feat)
        logits = self.head(pooled)
        fm = self._to_feature_map(feat)
        topk = logits.detach().float().topk(self.rr_regions, dim=1).indices
        w = self.head.cam_weight(fm.shape[1]).detach()
        rl, rf = [], []
        for i in range(self.rr_regions):
            cam = cam_map(fm.float(), w, topk[:, i])
            boxes = propose_regions(cam, self.rr_alpha)
            patch = crop_and_resize(x, boxes, self.rr_size, self.rr_margin)
            f = self.backbone.forward_features(patch)
            rf.append(self.backbone.forward_head(f))
            rl.append(self.head(rf[-1]))
        rl = torch.stack(rl, dim=1)                        # [B,K,C]
        rf = torch.stack(rf, dim=1)                        # [B,K,D]
        feats = routing_features(logits, rl, pooled, rf)
        fused, route_w = self.region_router.fuse(logits, rl, feats)
        if not self.training:
            return fused
        return {"logits": fused, "logits_region": rl, "route_w": route_w.detach()}

    def _pa_veto(self, logits: torch.Tensor) -> torch.Tensor:
        """一致否决：K 个部件全体同意的类与主分支不同时，按置信度决定是否改判。"""
        pa = getattr(self.backbone, "part_attn", None)
        lp = getattr(pa, "_last_part_logits", None) if pa is not None else None
        if lp is None:
            return logits
        with torch.no_grad():
            pp = lp.float().softmax(-1)                     # [B,K,C]
            vote = pp.argmax(-1).mode(dim=1).values          # [B]
            k = pp.shape[1]
            agree = (pp.argmax(-1) == vote[:, None]).all(dim=1)
            conf_v = pp.gather(2, vote[:, None, None].expand(-1, k, 1)).mean(1).squeeze(-1)
            conf_m = logits.float().softmax(-1).max(1).values
            if self.pa_veto_tau > 0:
                use = agree & (vote != logits.argmax(1)) & (conf_m < self.pa_veto_tau)
            else:
                use = agree & (vote != logits.argmax(1)) & (conf_v > conf_m)
        out = logits.clone()
        if use.any():
            out[torch.arange(out.shape[0], device=out.device)[use], vote[use]] += 1e3
        return out

    def _pa_logit_fuse(self, logits: torch.Tensor) -> torch.Tensor:
        """把 K 个部件描述子用主分类器的对应通道切片算 logits，按 β_k 加回主 logits。"""
        pa = getattr(self.backbone, "part_attn", None)
        if pa is None or getattr(pa, "_last_z", None) is None:
            return logits
        z = pa._last_z                                   # [B,K,g]
        W = self.head.fc.weight                          # [C_out, C]
        b = self.head.fc.bias
        # 部件分支如果用了通道重排（pa_perm），它的第 k 组对应原始通道 perm[k*g:(k+1)*g]；
        # 这里必须把分类权重按同样的顺序重排，否则特征与权重通道错位（审计 2026-10-29）。
        if getattr(pa, "perm", None) is not None:
            W = W[:, pa.perm]
        k = z.shape[1]
        c = W.shape[1]
        if c % k != 0 or z.shape[-1] != c // k:
            return logits                                # 非分组结构不适用
        Wk = W.view(W.shape[0], k, c // k)               # [C_out,K,g]
        lp = torch.einsum("bkg,okg->bko", z, Wk) + b[None, None, :]
        pa._last_part_logits = lp                        # 与主头共享权重
        beta = self.pa_beta[:k].view(1, k, 1)
        return logits + (beta * lp).sum(dim=1)

    def _forward_hyp_fuse(self, x: torch.Tensor):
        """整图 + 两个类别假设各自的支持区域（CAM 框）重编码后的融合。三次前向。"""
        feat = self.backbone.forward_features(x)
        logits = self.head(self.backbone.forward_head(feat))
        fm = self._to_feature_map(feat)
        top2 = logits.detach().float().topk(2, dim=1).indices
        w = self.head.cam_weight(fm.shape[1]).detach()
        outs = []
        for j in range(2):
            cam = cam_map(fm.float(), w, top2[:, j])
            boxes = mask_to_box(peak_mask(cam, self.hyp_alpha).detach())
            patch = crop_and_resize(x, boxes, self.hyp_size, self.hyp_margin)
            outs.append(self.head(self.backbone.forward_head(
                self.backbone.forward_features(patch))))
        if self.hyp_combiner:
            t = torch.stack([logits, outs[0], outs[1]], dim=1)      # [B,3,C]
            h = self.hyp_comb.proj(t) + self.hyp_comb.emb
            h = self.hyp_comb.blocks(h)
            fused = logits + self.hyp_comb.out(self.hyp_comb.norm(h).mean(dim=1))
        else:
            fused = logits + self.hyp_beta1 * outs[0] + self.hyp_beta2 * outs[1]
        if not self.training:
            return fused
        return {"logits": fused, "logits_hyp1": outs[0], "logits_hyp2": outs[1]}

    def _forward_part_rerank(self, x: torch.Tensor):
        """部件裁出来重新编码 + 部件间自注意力融合（MA-CNN 的流水线，融合器换成注意力）。

        测试 K+1 次前向：一次整图、K 次部件裁剪（放大到 pr_size）。
        训练与测试走同一条路径，部件分类辅助损失用独立的 part_head。
        """
        pa = getattr(self.backbone, "part_attn", None)
        if pa is None:
            raise RuntimeError("MODEL.part_rerank 需要同时开 MODEL.part_attn（提供部件掩码）")
        feat = self.backbone.forward_features(x)
        logits = self.head(self.backbone.forward_head(feat))
        div = pa._last_div
        masks = pa.part_masks(feat)                       # [B,K,h,w]
        b, k = masks.shape[0], masks.shape[1]
        if 0 < self.pr_parts < k:
            mass = masks.flatten(2).sum(-1)               # [B,K]
            idx = mass.topk(self.pr_parts, dim=1).indices
            masks = torch.gather(masks, 1, idx[..., None, None].expand(-1, -1, *masks.shape[2:]))
            k = self.pr_parts
        mk = masks.reshape(b * k, *masks.shape[2:])          # peak_mask 吃 [N,h,w]
        boxes = mask_to_box(peak_mask(mk, self.pr_alpha).detach()).view(b, k, 4)
        # NTS-Net 的 hard NMS 思想：两个框太像时，把后一个换成围绕同中心的外扩框，
        # 保证裁出来的 K 块不是同一块。tau 很小时框本来就分开，这里只是兜底。
        if k > 1:
            kept = []
            for i in range(k):
                cur = boxes[:, i]                              # [B,4]
                if kept:
                    prev = torch.stack(kept, dim=1)            # [B,i,4]
                    ix0 = torch.maximum(prev[..., 0], cur[:, None, 0])
                    iy0 = torch.maximum(prev[..., 1], cur[:, None, 1])
                    ix1 = torch.minimum(prev[..., 2], cur[:, None, 2])
                    iy1 = torch.minimum(prev[..., 3], cur[:, None, 3])
                    inter = (ix1 - ix0).clamp_min(0) * (iy1 - iy0).clamp_min(0)
                    area_c = (cur[:, 2] - cur[:, 0]) * (cur[:, 3] - cur[:, 1])
                    area_p = (prev[..., 2] - prev[..., 0]) * (prev[..., 3] - prev[..., 1])
                    iou = inter / (area_c[:, None] + area_p - inter).clamp_min(1e-6)
                    dup = (iou > 0.6).any(dim=1)               # [B]
                    cx = (cur[:, 0] + cur[:, 2]) / 2
                    cy = (cur[:, 1] + cur[:, 3]) / 2
                    hw = (cur[:, 2] - cur[:, 0]).clamp_min(0.05) * 0.9
                    hh = (cur[:, 3] - cur[:, 1]).clamp_min(0.05) * 0.9
                    wide = torch.stack([(cx - hw).clamp(0, 1), (cy - hh).clamp(0, 1),
                                        (cx + hw).clamp(0, 1), (cy + hh).clamp(0, 1)], dim=-1)
                    cur = torch.where(dup[:, None], wide, cur)
                kept.append(cur)
            boxes = torch.stack(kept, dim=1)                    # [B,K,4]
        parts = []
        for i in range(k):
            patch = crop_and_resize(x, boxes[:, i], self.pr_size, self.pr_margin)
            parts.append(self.backbone.forward_head(self.backbone.forward_features(patch)))
        pp = torch.stack(parts, dim=1)                    # [B,K,C]
        fused = logits + self.head(self.part_fusion(pp))
        if self.pr_use_cam:
            fm = self._to_feature_map(feat)
            pred = logits.detach().float().argmax(dim=1)
            w = self.head.cam_weight(fm.shape[1]).detach()
            cam = cam_map(fm.float(), w, pred)
            boxes_c = mask_to_box(peak_mask(cam, self.pr_alpha).detach())
            patch_c = crop_and_resize(x, boxes_c, self.pr_size, self.pr_margin)
            lg_c = self.head(self.backbone.forward_head(
                self.backbone.forward_features(patch_c)))
            fused = fused + self.pr_cam_beta * lg_c
        if not self.training:
            return fused
        return {"logits": fused, "logits_pa": self.part_head(pp), "pa_div": div}

    def _forward_cam_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """测试路径：整图 + CAM 框放大两路，按学到的 β 融合。两次前向。

        只看预测类算 CAM（测试时没有标签），与训练时 cam_aux 的做法一致。
        不走 model(patch)，避免递归触发本函数。
        """
        feat = self.backbone.forward_features(x)
        logits = self.head(self.backbone.forward_head(feat))
        fm = self._to_feature_map(feat)
        pred = logits.float().argmax(dim=1)
        w = self.head.cam_weight(fm.shape[1]).detach()
        cam = cam_map(fm.float(), w, pred)
        boxes = mask_to_box(peak_mask(cam, self.cam_aux.alpha_crop).detach())
        if self.cam_gate_area > 0:
            wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0)
            area = wh[:, 0] * wh[:, 1]                                # [B]
            gate = (area > self.cam_gate_area).to(logits.dtype).unsqueeze(-1)   # [B,1]
        else:
            gate = 1.0
        patch = crop_and_resize(x, boxes, self.cam_aux.out_size, self.cam_aux.margin)
        feat_r = self.backbone.forward_features(patch)
        logits_r = self.head(self.backbone.forward_head(feat_r))
        fused = logits + self.cam_beta * logits_r * gate
        if self.cam_fuse_regions >= 2 and self.cam_aux.margin2 is not None:
            patch2 = crop_and_resize(x, boxes, self.cam_aux.out_size, self.cam_aux.margin2)
            feat_r2 = self.backbone.forward_features(patch2)
            logits_r2 = self.head(self.backbone.forward_head(feat_r2))
            fused = fused + self.cam_beta2 * logits_r2
        return fused

    def _to_feature_map(self, feat: torch.Tensor) -> torch.Tensor:
        """把 backbone 的输出统一成 [B,C,h,w]，供 CAM / 部位模块使用。

        CNN（ResNet 等）本来就是 4 维，直接返回。
        ViT 返回的是 token 序列 [B,N,C]（可能含 cls token），这里按 patch 网格还原。
        只有 CAM 辅助分支需要这一步，主分类路径仍然用原始的 forward_head。
        """
        if feat.dim() == 4:
            return feat
        if feat.dim() != 3:
            raise ValueError(f"无法解释的特征形状 {tuple(feat.shape)}")
        b, n, c = feat.shape
        gh = gw = None
        pe = getattr(self.backbone, "patch_embed", None)
        grid = getattr(pe, "grid_size", None) if pe is not None else None
        if grid is not None:
            gh, gw = int(grid[0]), int(grid[1])
        if gh is not None and n == gh * gw:
            toks = feat
        elif gh is not None and n == gh * gw + 1:
            toks = feat[:, 1:]                       # 去掉 cls token
        else:
            for g in (int(round((n - 1) ** 0.5)), int(round(n ** 0.5))):
                if g * g == n - 1:
                    toks, gh, gw = feat[:, 1:], g, g
                    break
                if g * g == n:
                    toks, gh, gw = feat, g, g
                    break
            else:
                raise ValueError(f"token 数 {n} 不是平方数，无法还原成特征图")
        return toks.transpose(1, 2).reshape(b, c, gh, gw)

    def _forward_with_attn(self, x: torch.Tensor) -> torch.Tensor:
        """骨干 -> (池化特征 + attn 残差) -> 分类头。

        关键：y=None 的早退路径（CAM 裁剪分支、强视图）也必须走这里，
        否则那些分支拿到的特征不含 attn，而主分支含 attn ——
        裁剪一致性就会变成「把带模块的主分支拉向不带模块的裁剪分支」，
        等于给模块加了一层额外正则，配方就不再同源。
        """
        if self.attn_from == "layer3":
            f0, inter = self.backbone.forward_intermediates(x, indices=[3])
            src = inter[0]
        else:
            f0 = self.backbone.forward_features(x)
            src = self._to_feature_map(f0)
        pooled = f0 if getattr(self.head, "on_feature_map", False) else self.backbone.forward_head(f0)
        if self.attn_replace:
            return self.head(self.attn_pool(src))
        return self.head(pooled + self.attn_pool(src))

    def _forward_with_gate(self, x: torch.Tensor) -> torch.Tensor:
        """骨干 -> 通道门控 -> 池化 -> 分类头。用于 eval 与 y=None 的早退路径。"""
        f0 = self.backbone.forward_features(x)
        feat = self._to_feature_map(f0)
        feat = self.se_gate(feat)
        pooled = feat if getattr(self.head, "on_feature_map", False) else self.backbone.forward_head(feat)
        return self.head(pooled)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """返回池化后的特征 [B, feat_dim]；head 需要特征图时返回 [B, C, h, w]。"""
        if getattr(self.head, "on_feature_map", False):
            return self.backbone.forward_features(x)     # NetVLAD 等吃空间特征
        if self.fpn is None:
            return self.backbone(x)
        # layer3 是 28×28（浅、分辨率高），final 是 layer4 的 14×14（深、语义强）
        final, inter = self.backbone.forward_intermediates(x, indices=[3])
        fused = self.fpn(inter[0], final)            # [B, C, 28, 28]
        return self.backbone.forward_head(fused)     # GAP + flatten

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None,
                batch: dict | None = None):
        """评估时返回 logits [B, num_classes]；训练时若开了辅助分支，返回 dict。

        不在这里做 softmax —— CrossEntropyLoss 内部已经包含了。

        y 是类别标签，只有 CAM 辅助分支需要它（决定算哪一类的 CAM）。
        评估时不传 y，走的路径和以前完全一样。
        """
        # 三流融合：整图 / 头部 / 身体各过一次**同一个** backbone，特征拼接后分类。
        # 部位图由 dataset 按 GT keypoint 裁出（use_parts=True）。
        if self.fusion == "head_body":
            feats = [self.backbone(x)]
            if batch is not None:
                for k in ("head", "body"):
                    if k in batch:
                        feats.append(self.backbone(batch[k]))
            return self.head(torch.cat(feats, dim=1))

        # FPN 模式下只走主分支：辅助分支需要 layer4 的 [B,C,h,w]，
        # 而融合后的特征图语义不同，两者混在一起分不清是谁的贡献。
        if self.fpn is not None:
            return self.head(self.forward_features(x))

        # 双路融合（整图 + CAM 放大）：测试时走这条，两次前向
        if self.cam_fuse and not self.training:
            return self._forward_cam_fuse(x)

        # manifold mixup：batch 里带着「配对 + 比例」时，在池化后的特征上插值。
        # 只支持非 on_feature_map 的头（线性头）—— 那正是默认配置。
        if (self.training and batch is not None and "mix_perm" in batch
                and not getattr(self.head, "on_feature_map", False)):
            feat_map = self.backbone.forward_features(x)
            pa_cap = self._capture_parts()
            pooled = self.backbone.forward_head(feat_map)
            perm = batch["mix_perm"].to(pooled.device)
            lam = batch["mix_lam"].to(pooled.device).view(-1, 1)
            pooled = lam * pooled + (1.0 - lam) * pooled[perm]
            logits = self.head(pooled)
            # 开了 CAM 辅助分支时，裁剪仍走原图（它是另一种视角，不需要再混）
            if (self.cam_aux is not None or self.pair_aux is not None
                    or self.attn_aux is not None or pa_cap is not None) and y is not None:
                out = {"logits": logits}
                if pa_cap is not None:
                    out["logits_pa"] = pa_cap[0]
                    out["pa_div"] = pa_cap[1]
                if self.cam_aux is not None:
                    out.update(self.cam_aux(self, x, feat_map, logits, y, batch))
                    if self.cam_fuse and "logits_crop" in out:
                        gate = 1.0
                        if self.cam_gate_area > 0 and "cam_box_area" in out:
                            gate = (out["cam_box_area"] > self.cam_gate_area).to(
                                out["logits"].dtype).unsqueeze(-1)
                        fused = out["logits"] + self.cam_beta * out["logits_crop"] * gate
                        if "logits_crop2" in out and getattr(self, "cam_beta2", None) is not None:
                            fused = fused + self.cam_beta2 * out["logits_crop2"]
                        out["logits_fused"] = fused
                if self.attn_aux is not None:
                    out["logits_parts"] = self.attn_aux(feat_map)
                if self.pair_aux is not None:
                    out.update(self.pair_aux(self, pooled, y))
                return out
            return logits

        if self.region_route:
            return self._forward_region_route(x)
        if self.hyp_fuse:
            return self._forward_hyp_fuse(x)
        if self.part_rerank:
            return self._forward_part_rerank(x)

        # PMG 走独立路径：它自己就要四次 forward，和 CAM 辅助分支不兼容
        if self.tokenlearner is not None and not self.training:
            f0 = self.backbone.forward_features(x)
            feat = self._to_feature_map(f0)
            if self.tl_replace:
                return self.head(self.tokenlearner(feat))
            return self.head(self.backbone.forward_head(f0) + self.tokenlearner(feat))
        if self.attn_pool is not None and not self.training:
            return self._forward_with_attn(x)
        if self.pmg is not None and not self.training:
            # 推理只用原图那一路，走 concat 分类器（单模型单前向）
            fin, inter = self.backbone.forward_intermediates(x, indices=[2, 3])
            return self.pmg(inter[0], inter[1], fin)["logits"]
        if self.pmg is not None and self.training:
            outs = []
            for xx in (jigsaw(x, 8), jigsaw(x, 4), jigsaw(x, 2), x):
                fin, inter = self.backbone.forward_intermediates(xx, indices=[2, 3])
                # inter[0]=layer2(512,56²)  inter[1]=layer3(1024,28²)  fin=layer4(2048,14²)
                outs.append(self.pmg(inter[0], inter[1], fin))
            return {
                "logits": outs[3]["logits"],          # 原图 + concat 分类器（推理用这一路）
                "logits_pmg1": outs[0]["logits_pmg1"],
                "logits_pmg2": outs[1]["logits_pmg2"],
                "logits_pmg3": outs[2]["logits_pmg3"],
            }

        use_cam = self.cam_aux is not None
        # CutMix 把画面抹掉一块，CAM 会框到残留的碎片上，裁出来的东西没有语义。
        # 这批直接跳过 CAM 分支（MixUp 的 batch 照常做，叠影不破坏结构）。
        if batch is not None and batch.get("mix_mode") == "cutmix":
            use_cam = False
        use_attn = self.attn_aux is not None
        use_pair = self.pair_aux is not None
        # 强视图由 trainer 在 batch 里塞进来（张量级几何扰动，见 dataset/strong_view.py）
        use_strong = batch is not None and "image_strong" in batch
        use_ds = self.deep_sup is not None
        use_ap = self.attn_pool is not None
        use_tl = self.tokenlearner is not None
        use_se = self.se_gate is not None
        # 部件模块即使没有别的辅助分支，训练时也需要 dict 输出（逐部件 CE 与散度项）
        use_pa = getattr(self.backbone, "part_attn", None) is not None
        if not (use_cam or use_attn or use_pair or use_strong or use_ds or use_ap
                or use_se or use_tl or use_pa) or not self.training or y is None:
            if self.attn_pool is not None:
                return self._forward_with_attn(x)      # CAM 裁剪/强视图也带上 attn
            if self.se_gate is not None:
                return self._forward_with_gate(x)      # 同理，门控也要生效
            if self.tokenlearner is not None:
                f0 = self.backbone.forward_features(x)
                feat = self._to_feature_map(f0)
                if self.tl_replace:
                    return self.head(self.tokenlearner(feat))
                return self.head(self.backbone.forward_head(f0) + self.tokenlearner(feat))
            if self.pa_logit_fuse:
                f0 = self.backbone.forward_features(x)
                lg = self.head(self.backbone.forward_head(f0))
                return self._pa_logit_fuse(lg)
            if self.pa_veto:
                f0 = self.backbone.forward_features(x)
                lg = self.head(self.backbone.forward_head(f0))
                return self._pa_veto(lg)
            return self.head(self.forward_features(x))

        # 池化前 / 池化后拆开跑一次，避免 backbone 跑两遍
        need_l3 = (self.deep_sup is not None) or (
            self.attn_pool is not None and self.attn_from == "layer3")
        if need_l3:
            # 需要 layer3：一次前向同时拿 intermediate 和 final，不重复跑骨干
            feat, inter = self.backbone.forward_intermediates(x, indices=[3])
            if self.deep_sup is not None:
                self._ds_feat = inter[0]
            if self.attn_pool is not None and self.attn_from == "layer3":
                self._ap_feat = inter[0]
        else:
            feat = self.backbone.forward_features(x)      # CNN: [B,C,h,w]；ViT: [B,N,C]
        # 吃特征图的头（gapstd / netvlad / cap）不能再过 forward_head，
        # 否则维度从 [B,C,h,w] 变成 [B,C]，头会直接报错。
        pooled = (feat if getattr(self.head, "on_feature_map", False)
                  else self.backbone.forward_head(feat))
        feat = self._to_feature_map(feat)                 # 下游 CAM/部位模块统一吃 4 维
        # 部件注意力的描述子必须在**主干第一次前向之后、CAM 裁剪分支之前**取走：
        # 裁剪分支会再跑一次主干，把 _last_z 覆盖成裁剪图上的部件。
        pa_cap = self._capture_parts()
        if self.se_gate is not None:
            feat = self.se_gate(feat)
            pooled = self.backbone.forward_head(feat) if not getattr(
                self.head, "on_feature_map", False) else feat
        if self.attn_pool is not None:
            src = getattr(self, "_ap_feat", None) if self.attn_from == "layer3" else feat
            pooled = self.attn_pool(src) if self.attn_replace else pooled + self.attn_pool(src)
        if self.tokenlearner is not None:
            pooled = self.tokenlearner(feat) if self.tl_replace else pooled + self.tokenlearner(feat)
            if self.tokenlearner._reg is not None:
                out_reg = self.tokenlearner._reg
        mge_logits = self.mge() if self.mge is not None else None
        logits = self.head(pooled)
        if self.pa_logit_fuse:
            logits = self._pa_logit_fuse(logits)
        if self.pa_veto and not self.training:
            logits = self._pa_veto(logits)

        out = {"logits": logits}
        if mge_logits is not None:
            out["logits_mge"] = mge_logits
        if self.pa_logit_fuse and getattr(self.backbone, "part_attn", None) is not None:
            # logit 级融合时，逐部件 CE 用**权重绑定**的那套 logits（_pa_logit_fuse 里写回），
            # 而不是模块自带的独立 part_cls —— 否则两套头并存，又回到「证据与出口解耦」。
            tied = getattr(self.backbone.part_attn, "_last_part_logits", None)
            if tied is not None:
                out["logits_pa"] = tied
                out["pa_div"] = self.backbone.part_attn._last_div
        if pa_cap is not None and "logits_pa" not in out:
            # 每个部件单独过同一个分类头（MA-CNN 的部件损失）：迫使每个部件都带类别信息，
            # 否则检测器只靠最终融合特征的梯度，很容易退化成"全都在看同一处"。
            out["logits_pa"] = pa_cap[0]
            out["pa_div"] = pa_cap[1]
        if self.tokenlearner is not None and self.tokenlearner._reg is not None:
            out["tl_reg"] = self.tokenlearner._reg
        if self.training and getattr(self.tokenlearner, "_last_z", None) is not None:
            # 逐 token 辅助 CE（MA-CNN 式部件损失）：每个学到的部件 token 都要单独
            # 认对类别。它把「往背景上飘」这条路堵死——背景特征不可能单独分类；
            # 同时不阻止 token 之间分化（那是 JS 项的职责），两者互补。
            # 先展平成 [B*S,C] 再过分类头：head 可能带 BatchNorm1d，直接喂 [B,S,C]
            # 会把 S 当成通道维、静默算错（head_norm=true 时）。
            z = self.tokenlearner._last_z                    # [B,S,C]
            bz, sz, cz = z.shape
            out["logits_tl"] = self.head(z.reshape(bz * sz, cz)).reshape(bz, sz, -1)
        if self.part_train_aux and self.training and y is not None:
            pa = getattr(self.backbone, "part_attn", None)
            if pa is not None:
                b0 = feat.shape[0]
                if self.pta_use_cam:
                    fm = self._to_feature_map(feat)
                    top = logits.detach().float().topk(self.pta_parts, dim=1).indices
                    w = self.head.cam_weight(fm.shape[1]).detach()
                    boxes = []
                    for i in range(top.shape[1]):
                        cam = cam_map(fm.float(), w, top[:, i])
                        boxes.append(mask_to_box(peak_mask(cam, self.pta_cam_alpha).detach()))
                    bx = torch.stack(boxes, dim=1)                # [B,n,4]
                    n_use = bx.shape[1]
                else:
                    masks = pa.part_masks(feat)                   # [B,K,h,w]
                    k0 = masks.shape[1]
                    n_use = min(self.pta_parts, k0)
                    mk = masks.reshape(b0 * k0, *masks.shape[2:])
                    bx = mask_to_box(peak_mask(mk, self.pta_alpha).detach()).view(b0, k0, 4)
                pl = []
                for i in range(n_use):
                    patch = crop_and_resize(x, bx[:, i], self.pr_size, self.pta_margin)
                    pl.append(self.head(self.backbone.forward_head(
                        self.backbone.forward_features(patch))))
                lp = torch.stack(pl, dim=1)                        # [B,n,C]
                out["logits_partaux"] = lp
                if self.pta_cons_w > 0:
                    import torch.nn.functional as F
                    T = self.pta_cons_T
                    lg = F.log_softmax(logits.detach().float() / T, dim=1).unsqueeze(1)
                    lpp = F.log_softmax(lp.float() / T, dim=-1)
                    out["part_cons"] = (lg.exp() * (lg - lpp)).sum(-1).mean()
        if use_cam:
            out.update(self.cam_aux(self, x, feat, logits, y, batch))
            if self.cam_fuse and "logits_crop" in out:
                gate = 1.0
                if self.cam_gate_area > 0 and "cam_box_area" in out:
                    gate = (out["cam_box_area"] > self.cam_gate_area).to(
                        out["logits"].dtype).unsqueeze(-1)
                fused = out["logits"] + self.cam_beta * out["logits_crop"] * gate
                if "logits_crop2" in out and getattr(self, "cam_beta2", None) is not None:
                    fused = fused + self.cam_beta2 * out["logits_crop2"]
                out["logits_fused"] = fused
        if use_attn:
            out["logits_parts"] = self.attn_aux(feat)     # [B, K, num_classes]
        # 成对判别：吃池化后的特征，自己选出批内最近邻，再生成门控后的四路 logits。
        # 用的是 forward_head 后的 pooled（不是池化前的 feat），所以那几种吃特征图的头
        # （gapstd / netvlad / cap）不支持 —— build_model 里已经拦掉了。
        if self.deep_sup is not None:
            fds = self._ds_feat.mean(dim=(2, 3)) if self._ds_feat.dim() == 4 else self._ds_feat.mean(1)
            out["logits_ds"] = self.deep_sup(fds)
        if use_pair:
            out.update(self.pair_aux(self, pooled, y))
        # 强视图只出 logits，不触发 cam/attn/pair（否则会递归多跑一路 CAM 分支）
        if use_strong:
            out["logits_strong"] = self.head(self.forward_features(batch["image_strong"]))
        return out


def build_model(cfg: dict) -> Classifier:
    """按 config 建模型。

    num_classes 从 DATASET.num_classes 注入，不写在 MODEL.kwargs 里 ——
    避免同一个数字在配置里出现两次、改一处忘一处。
    """
    mcfg = cfg["MODEL"]
    bkwargs = dict(mcfg.get("kwargs", {}))
    if mcfg.get("dynamic_img_size", False) and "vit" in mcfg["name"]:
        bkwargs.setdefault("dynamic_img_size", True)   # 让 ViT 支持 448
    backbone = build_backbone(mcfg["name"], pool=mcfg.get("pool", "avg"),
                              gem_p=float(mcfg.get("gem_p", 3.0)),
                              se=bool(mcfg.get("se", False)),
                              stride8=bool(mcfg.get("stride8", False)),
                              content_pool=bool(mcfg.get("content_pool", False)),
                              content_rank=int(mcfg.get("content_rank", 64)),
                              **bkwargs)

    # 冻结骨干（只训练部件分支/头）：部件变成「事后附加」的模块，
    # 主分支与基线逐位相同，否决票才是真正零代价的。
    if mcfg.get("freeze_backbone", False):
        # 骨干（除部件分支）与分类头都冻结：主分支逐位等于基线，
        # 只有部件检测器 + 部件分类头在训练。
        for n, prm in backbone.named_parameters():
            if not n.startswith("part_attn"):
                prm.requires_grad = False
    # 特征维度从模型自身读，不写死
    feat_dim = backbone.num_features
    # 可选：在 layer3 与 layer4 之间插一层全局自注意力（见 model/global_ctx.py）。
    # 挂在 backbone 上，所以会被 .parameters()/state_dict 正常收进去。
    build_global_context(backbone, mcfg)
    # 可选：部件提取 + 部件间自注意力（见 model/part_attn.py），挂在 layer4 的
    # forward hook 上，GAP 之前把结果加进去。
    build_part_attention(backbone, mcfg, int(cfg["DATASET"]["num_classes"]))

    head_name = mcfg.get("head", "linear")
    if head_name not in HEADS:
        raise KeyError(f"未知的 head: {head_name!r}，可选 {sorted(HEADS)}")
    head_kwargs = dict(
        in_features=feat_dim,
        out_features=cfg["DATASET"]["num_classes"],
        # 注意作用面很小：这个头只有 feat_dim × num_classes 个参数（ResNet50 上是 410K，
        # backbone 是 23.5M）。想正则 backbone 得用 MODEL.kwargs.drop_path_rate。
        dropout=mcfg.get("dropout", 0.0),
    )
    # 线性头前面加一层 BN，把不同预训练权重的特征尺度拉齐（见 LinearHead 的说明）
    if head_name == "linear":
        head_kwargs["norm"] = bool(mcfg.get("head_norm", False))
    # NetVLAD 专有参数：簇数，以及 assign 卷积的初始化放大倍数
    # （默认初始化会让 softmax 直接饱和成均匀，32 个簇全冗余，见 model/head.py）
    if head_name == "netvlad":
        head_kwargs["num_clusters"] = int(mcfg.get("netvlad_clusters", 32))
        head_kwargs["assign_init_scale"] = float(mcfg.get("netvlad_init_scale", 60.0))
    if head_name == "cap":
        head_kwargs["lstm_hidden"] = int(mcfg.get("cap_lstm", 128))
        head_kwargs["clusters"] = int(mcfg.get("cap_clusters", 32))
        head_kwargs["use_vlad"] = bool(mcfg.get("cap_vlad", False))
        head_kwargs["region"] = str(mcfg.get("cap_region", "attn"))
    if head_name == "cbp":
        head_kwargs["dim"] = int(mcfg.get("cbp_dim", 8192))
    head = HEADS[head_name](**head_kwargs)
    if mcfg.get("freeze_backbone", False):
        for prm in head.parameters():
            prm.requires_grad = False

    fusion = mcfg.get("fusion", "")
    cam_aux = build_cam_aux(cfg["TRAIN"], cfg["DATASET"]["image_size"])
    attn_aux = build_attention_aux(cfg["TRAIN"], mcfg, feat_dim,
                                   cfg["DATASET"]["num_classes"])
    fpn = build_fpn(backbone, mcfg)
    # head 的输入维度：FPN 融合后是 FPN 输出通道；三流融合是 feat_dim × 3
    in_dim = feat_dim
    if fpn is not None:
        in_dim = fpn.out_channels
    elif fusion == "head_body":
        in_dim = feat_dim * 3
    if in_dim != feat_dim:
        # 重建时必须带上原来那份 kwargs，否则 head_norm / netvlad / cap / cbp 的
        # 专有参数会被静默丢掉（例如 fpn + head_norm=true 时模型里其实没有 BN）
        head_kwargs = dict(head_kwargs)
        head_kwargs["in_features"] = in_dim
        head = HEADS[head_name](**head_kwargs)
    if mcfg.get("freeze_backbone", False):
        for prm in head.parameters():
            prm.requires_grad = False

    # 成对判别（API-Net）。它吃的是 forward_head 之后的池化向量，
    # 所以和三流融合、FPN、吃特征图的头都不兼容 —— 这几种情况直接报错，
    # 不要静默降级成“开了但没生效”。
    pair_aux = None
    if mcfg.get("pair", False):
        if fusion == "head_body" or fpn is not None:
            raise ValueError("MODEL.pair 不支持与 MODEL.fusion / MODEL.fpn 同时开启"
                             "（成对判别用的是池化后的向量）")
        if getattr(head, "on_feature_map", False):
            raise ValueError(f"MODEL.pair 不支持吃特征图的头（head={head_name}）")
        pair_aux = PairAux(feat_dim, hidden=int(mcfg.get("pair_hidden", 512)))

    se_gate = SEGate(feat_dim) if mcfg.get("se_gate", False) else None

    tokenlearner = None
    if mcfg.get("tokenlearner", False):
        tokenlearner = TokenLearner(
            in_channels=feat_dim, out_dim=feat_dim,
            dim=int(mcfg.get("tl_dim", 256)),
            s_tokens=int(mcfg.get("tl_tokens", 8)),
            # TL 自己的 dropout（留空沿用 MODEL.dropout，保持历史口径）
            dropout=float(mcfg.get("tl_dropout") if mcfg.get("tl_dropout") is not None
                          else mcfg.get("dropout", 0.0)),
            ent_target=float(mcfg.get("tl_ent_target", 0.0)),
            ent_mode=str(mcfg.get("tl_ent_mode", "target")),
            ent_floor=float(mcfg.get("tl_ent_floor", 0.5)),
            div_w=float(mcfg.get("tl_div_w", 0.0)),
            js_w=float(mcfg.get("tl_js_w", 0.0)),
            js_target=float(mcfg.get("tl_js_target", 0.35)),
            js_reduce=str(mcfg.get("tl_js_reduce", "mean")),
            attn_depth=int(mcfg.get("tl_attn_depth", 0)),
            attn_heads=int(mcfg.get("tl_attn_heads", 4)),
            readout=str(mcfg.get("tl_readout", "mean")),
            replace=bool(mcfg.get("tl_replace", False)))

    attn_pool = None
    if mcfg.get("attn_pool", False):
        # 兼容性断言：这几种组合要么训练/推理不同源，要么会静默忽略模块
        if mcfg.get("pmg", False) or fpn is not None or fusion == "head_body":
            raise ValueError("MODEL.attn_pool 不能与 pmg / fpn / fusion 同时开启")
        if getattr(head, "on_feature_map", False):
            raise ValueError(f"MODEL.attn_pool 不支持吃特征图的头（head={head_name}）")
        _toks, _from = int(mcfg.get("attn_tokens", 196)), str(mcfg.get("attn_from", "layer4"))
        _expect = {("layer4", 196), ("layer4", 49), ("layer3", 784)}
        if (_from, _toks) not in _expect:
            raise ValueError(
                f"token 数与特征图不匹配：attn_from={_from} 只支持 "
                f"{[t for f, t in _expect if f == _from]}，收到 {_toks}")
        attn_pool = AttnPool(
            in_channels=(1024 if mcfg.get("attn_from", "layer4") == "layer3" else feat_dim),
            out_dim=feat_dim,
            dim=int(mcfg.get("attn_dim", 256)),
            tokens=int(mcfg.get("attn_tokens", 196)),
            depth=int(mcfg.get("attn_depth", 2)),
            heads=int(mcfg.get("attn_heads", 4)),
            use_pos=bool(mcfg.get("attn_pos", False)),
            readout=str(mcfg.get("attn_readout", "mean")),
            replace=bool(mcfg.get("attn_replace", False)),
            dropout=float(mcfg.get("attn_dropout", mcfg.get("dropout", 0.0))))

    deep_sup = None
    if mcfg.get("deep_sup", False):
        deep_sup = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, cfg["DATASET"]["num_classes"]))

    pmg = None
    if mcfg.get("pmg", False):
        # PMG 自己每步四次前向，叠 CAM 分支会变成六次；官方也不用 CAM。
        # 这里强制关掉，避免有人同时开。
        cam_aux = None
        pmg = PMGHead(512, 1024, 2048, cfg["DATASET"]["num_classes"],
                      feature_size=int(mcfg.get("pmg_feature_size", 512)))

    cam_fuse = bool(mcfg.get("cam_fuse", False))
    cam_fuse_regions = int(mcfg.get("cam_fuse_regions", 1))
    if cam_fuse and cam_aux is None:
        raise ValueError("MODEL.cam_fuse 需要 TRAIN.cam_crop > 0（裁剪分支是融合的另一路）")
    if cam_fuse and cam_fuse_regions >= 2 and cam_aux.margin2 is None:
        raise ValueError("cam_fuse_regions>=2 需要 TRAIN.cam_margin2（第二路的裁剪外扩）")
    if cam_fuse and mcfg.get("fusion") == "head_body":
        raise ValueError("MODEL.cam_fuse 不支持与 MODEL.fusion 同时开启")
    if mcfg.get("region_route", False) and float(cfg["TRAIN"].get("cam_crop", 0.0)) > 0:
        raise ValueError("MODEL.region_route 会接管 forward 并跳过 CAM 分支；"
                         "请显式设 TRAIN.cam_crop=0（否则 CAM 损失静默不生效）")
    if (mcfg.get("pa_multiscale", False) and float(cfg["TRAIN"].get("pa_aux_w", 0.0)) > 0
            and not mcfg.get("pa_logit_fuse", False)):
        raise ValueError("MultiScaleParts 没有自己的部件分类头；开了 pa_aux_w 就必须开 "
                         "pa_logit_fuse，否则逐部件 CE 会静默缺失")
    part_rerank = bool(mcfg.get("part_rerank", False))
    if part_rerank and not mcfg.get("part_attn", False):
        raise ValueError("MODEL.part_rerank 需要同时开 MODEL.part_attn（提供部件掩码）")
    mge = build_mge(backbone, mcfg, cfg["DATASET"]["num_classes"])
    freeze = bool(mcfg.get("freeze_backbone", False))
    return Classifier(backbone, head, cam_aux, attn_aux, fpn, fusion, pair_aux, pmg,
                      deep_sup, attn_pool, str(mcfg.get("attn_from", "layer4")), se_gate,
                      tokenlearner, cam_fuse=cam_fuse,
                      cam_fuse_regions=cam_fuse_regions,
                      cam_gate_area=float(mcfg.get("cam_gate_area", 0.0)),
                      part_rerank=part_rerank,
                      pr_margin=float(mcfg.get("pr_margin", 1.3)),
                      pr_alpha=float(mcfg.get("pr_alpha", 0.2)),
                      pr_size=int(mcfg.get("pr_size", cfg["DATASET"]["image_size"])),
                      pr_parts=int(mcfg.get("pr_parts", 0)),
                      pr_use_cam=bool(mcfg.get("pr_use_cam", False)),
                      hyp_fuse=bool(mcfg.get("hyp_fuse", False)),
                      hyp_margin=float(mcfg.get("hyp_margin", 1.3)),
                      hyp_alpha=float(mcfg.get("hyp_alpha", 0.2)),
                      hyp_size=int(mcfg.get("hyp_size", cfg["DATASET"]["image_size"])),
                      hyp_combiner=bool(mcfg.get("hyp_combiner", False)),
                      pa_logit_fuse=bool(mcfg.get("pa_logit_fuse", False)),
                      pa_parts=int(mcfg.get("pa_parts", 4)),
                      pa_veto=bool(mcfg.get("pa_veto", False)),
                      pa_veto_tau=float(cfg["TRAIN"].get("pa_veto_tau", 0.0)),
                      freeze_backbone=freeze,
                      mge=mge,
                      region_route=bool(mcfg.get("region_route", False)),
                      rr_regions=int(mcfg.get("rr_regions", 2)),
                      rr_alpha=float(mcfg.get("rr_alpha", 0.2)),
                      rr_margin=float(mcfg.get("rr_margin", 1.3)),
                      rr_hidden=int(mcfg.get("rr_hidden", 64)),
                      rr_size=int(mcfg.get("rr_size", cfg["DATASET"]["image_size"])),
                      part_train_aux=bool(mcfg.get("part_train_aux", False)),
                      pta_parts=int(mcfg.get("pta_parts", 2)),
                      pta_margin=float(mcfg.get("pta_margin", 1.3)),
                      pta_alpha=float(mcfg.get("pta_alpha", 0.6)),
                      pta_cons_w=float(cfg["TRAIN"].get("pta_cons_w", 0.0)),
                      pta_cons_T=float(cfg["TRAIN"].get("pta_cons_T", 2.0)),
                      pta_use_cam=bool(mcfg.get("pta_use_cam", False)),
                      pta_cam_alpha=float(mcfg.get("pta_cam_alpha", 0.2)))


if __name__ == "__main__":
    cfg = {
        "DATASET": {"num_classes": 200},
        "MODEL": {"name": "resnet50", "head": "linear", "kwargs": {"pretrained": False}},
    }
    model = build_model(cfg)
    x = torch.randn(2, 3, 224, 224)
    print("forward_features:", tuple(model.forward_features(x).shape))  # [2, 2048]
    print("forward         :", tuple(model(x).shape))                   # [2, 200]
