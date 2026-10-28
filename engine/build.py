"""按 config 构建损失 / 优化器 / 学习率调度器。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationCriterion(nn.Module):
    """把 nn.CrossEntropyLoss 适配成 `criterion(outputs, batch)` 的签名。

    为什么要多包这一层：训练/评估循环里只认 `criterion(outputs, batch)` 这一个签名，
    将来换成多任务损失（分类 + 属性 + 部位）时，循环一行都不用改。
    现在多花这十行，是为了以后不用回头改循环。

    forward 返回 (loss, logs)，logs 是给日志用的额外标量（现在没有）。
    """

    def __init__(self, label_smoothing: float = 0.0,
                 aux_weights: dict[str, float] | None = None,
                 hard_label_aux: tuple[str, ...] = (),
                 consistency_w: float = 0.0, consistency_T: float = 2.0,
                 consistency2_w: float = 0.0,
                 pair_w: float = 0.0, pair_eps: float = 0.05,
                 pair_rank_w: float = 1.0, hard_neg_k: int = 0,
                 entropy_w: float = 0.0, main_w: float = 1.0, strong_ce: float = 0.0,
                 distill_w: float = 0.0, distill_T: float = 2.0,
                 distill_min_conf: float = 0.0, distill_gate_label: bool = False,
                 tl_reg_w: float = 0.0, pa_div_w: float = 0.0, part_cons_w: float = 0.0,
                 part_view_cons_w: float = 0.0,
                 strong_kl: float = 0.0, strong_T: float = 2.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # 成对判别（API-Net）的总权重；0 = 关闭。见 model/pair.py。
        self.pair_w = float(pair_w)
        # 排名铰链的 margin（原文 ε = 0.05）和系数（原文 λ = 1）
        self.pair_eps = float(pair_eps)
        self.pair_rank_w = float(pair_rank_w)
        # 硬负样本交叉熵（GCE，Sun et al. AAAI'20, 原始 conv 版 2019）。
        # 0 = 关闭，用普通 CE；k>0 时主分支改用「正类 + 得分最高的 k 个负类」的
        # softmax。动机见 EXPERIMENT_LOG 的诊断：错误是对称的类对混淆，
        # 真正竞争的是那一两个最像的负类，其余 198 个负类只贡献分母。
        self.hard_neg_k = int(hard_neg_k)
        # 熵正则（MaxEnt, Dubey et al. NeurIPS 2018）：最大化输出的熵。
        # 加在总损失上是减号，0 = 关闭。
        self.entropy_w = float(entropy_w)
        # 主分支损失的权重。PMG 用 2（官方 concat 权重），其余实验保持 1
        self.main_w = float(main_w)
        # 蒸馏：教师 logits 由 trainer 放进 batch，这里是学生追教师的 KL
        self.distill_w = float(distill_w)
        self.distill_T = float(distill_T)
        # 教师置信度门控：教师自己都没有把握的样本不蒸馏。
        # 依据是分解结果 —— 蒸馏的 −0.48 全部来自"教师错而学生对"的那批图。
        self.distill_min_conf = float(distill_min_conf)
        # 按标签门控：只在"教师的预测等于真实标签"的样本上蒸馏。
        # 训练标签是已知的，所以这不是泄漏；它去掉的正是"教师错而学生对"那批有害样本。
        self.distill_gate_label = bool(distill_gate_label)
        # TokenLearner 的防塌缩正则权重（熵目标 + 图间多样性），由 model 放进 outputs
        self.tl_reg_w = float(tl_reg_w)
        # 部件注意力：部件之间看同一处的惩罚（MA-CNN 的散度项）。模块内部算好可微的
        # 平均余弦重叠，这里只乘权重。
        self.pa_div_w = float(pa_div_w)
        self.part_cons_w = float(part_cons_w)
        # 额外部位视图与主分支的一致性（与 pta_cons_w 不是一回事，别混）
        self.part_view_cons_w = float(part_view_cons_w)
        # 强视图：CE 权重（1.0 = 只当更强的增强）与非对称 KL 权重（教师是主分支、detach）
        self.strong_ce = float(strong_ce)
        self.strong_kl = float(strong_kl)
        self.strong_T = float(strong_T)
        # 主分支和裁剪分支的分布一致性（裁剪一致性）。裁剪看到的是放大后的局部，
        # 强迫两条路径给出相同分布，等于让主干学会「从能被放大的那部分提取信息」，
        # 而测试时不给放大 —— 真正被强化的是在 448 下把细节榨出来的能力。
        self.consistency_w = float(consistency_w)
        # 第二视图（更大外扩的裁剪）与主分支的一致性，独立权重
        self.consistency2_w = float(consistency2_w)
        self.consistency_T = float(consistency_T)
        # 这些辅助分支用**硬标签**而不是 mixup 的 soft_target。
        # 理由：裁剪分支切的是「A 最显著的那块」，那块区域可能是纯 A 的像素
        # （CutMix 的贴入区在别处），要求它输出 0.7A+0.3B 是错误监督。
        self.hard_label_aux = set(hard_label_aux)
        # 各路辅助分支的损失权重，键要和 model 里产出的 key 对上：
        #   logits_drop / logits_crop  -> model/cam_aug.py
        #   logits_parts               -> model/attention.py
        self.aux_weights = {k: float(v) for k, v in (aux_weights or {}).items()}

    def _one(self, logits, batch, hard_label: bool = False):
        """一路分支的损失：Mixup 时用软目标，否则普通 CE。

        Mixup / CutMix 会在 batch 里塞一个 soft_target（见 dataset/mixup.py）。
        混出来的图同时属于两个类，硬标签不成立，得用软目标交叉熵；这时也不再叠
        label_smoothing —— 软目标本身已经把目标压平了。

        logits 也可以是 [B, K, num_classes]（注意力模块的 K 个部位），这时把 K 展开
        成一维的 B*K 个样本，每个部位都要独立认对。
        """
        K = 1
        if logits.dim() == 3:
            B, K, C = logits.shape
            logits = logits.reshape(B * K, C)

        use_soft = ("soft_target" in batch) and not hard_label
        label = batch["label"]
        if K > 1:
            label = label.repeat_interleave(K)

        # 硬负样本 CE：硬标签和 mixup 软目标都支持（见 _gce）。
        if self.hard_neg_k > 0:
            target = batch["soft_target"] if use_soft else None
            # K>1 时 logits 已展平成 B*K，软目标也要跟着展开，否则在 _gce 里
            # broadcast 会直接报错（attn_weight 与 hard_neg_k 同时开就会踩到）
            if target is not None and K > 1:
                target = target.repeat_interleave(K, dim=0)
            return self._gce(logits, label, target)

        if "soft_target" in batch and not hard_label:
            target = batch["soft_target"]
            if K > 1:
                target = target.repeat_interleave(K, dim=0)
            logp = F.log_softmax(logits, dim=1)
            return -(target * logp).sum(dim=1).mean()

        return self.ce(logits, label)

    def _gce(self, logits, label, target=None):
        """softmax 的分母只保留「正类 + 最难的那 k 个负类」。

        硬标签：L = -log( e^{s_l} / (e^{s_l} + Σ_{j∈J'} e^{s_j}) )，J' 是负类 top-k。
        mixup 软目标：正类是 target>0 的那几个类（通常是两个），
        负类同样只留 top-k，再在保留集合上做归一化。
        这样混合 batch 也照样施加压力，不必把 mixup 关掉。
        """
        logits = logits.float()
        k = self.hard_neg_k

        if target is None:
            s_l = logits.gather(1, label.view(-1, 1)).squeeze(1)
            neg = logits.scatter(1, label.view(-1, 1), float("-inf"))
            keep_k = min(k, neg.size(1) - 1)
            topk = neg.topk(keep_k, dim=1).values
            denom = torch.logsumexp(torch.cat([s_l.view(-1, 1), topk], dim=1), dim=1)
            return (denom - s_l).mean()

        support = target > 0
        neg = logits.masked_fill(support, float("-inf"))
        keep_k = max(1, min(k, logits.size(1) - int(support.sum(dim=1).max())))
        keep = support.scatter(1, neg.topk(keep_k, dim=1).indices, True)
        logp = F.log_softmax(logits.masked_fill(~keep, float("-inf")), dim=1)
        # -inf 乘 0 会出 nan，保留集合之外直接置 0
        logp = torch.where(keep, logp, torch.zeros_like(logp))
        return -(target * logp).sum(dim=1).mean()

    def _pair_loss(self, outputs, batch):
        """API-Net 的成对损失：四个增强特征的 CE + 排名铰链（原文 Eq.8-10）。

        为什么标签要在这里取而不是在 model 里取：mixup 下目标是软标签，
        模型那一层不该知道软目标长什么样（它只管产出特征和 logits）。

        四个特征里，self 是用自己的门控激活的，other 是用对家的门控激活的。
            L_CE = 四路交叉熵的平均
            L_rk = mean( relu(p_other[c] - p_self[c] + eps) )，i 和 j 各一项
        self 应该比 other 更认得出自己，铰链就是把这个优先级推开。
        """
        idx_i, idx_j = outputs["pair_idx_i"], outputs["pair_idx_j"]
        li = batch["label"][idx_i]
        lj = batch["label"][idx_j]

        if "soft_target" in batch:
            st = batch["soft_target"]
            ti, tj = st[idx_i], st[idx_j]
            # 排名损失要一个「正确类」的下标：软目标里权重最大的那个
            c_i, c_j = ti.argmax(1), tj.argmax(1)
        else:
            C = outputs["pair_self_i"].size(1)
            ti = F.one_hot(li, C).to(outputs["pair_self_i"].dtype)
            tj = F.one_hot(lj, C).to(outputs["pair_self_j"].dtype)
            c_i, c_j = li, lj

        ce = 0.0
        for key, t in (("pair_self_i", ti), ("pair_other_i", ti),
                       ("pair_self_j", tj), ("pair_other_j", tj)):
            ce = ce - (t * F.log_softmax(outputs[key].float(), dim=1)).sum(dim=1)
        ce = ce.mean() / 4.0

        def p(key, idx):
            return F.softmax(outputs[key].float(), dim=1).gather(
                1, idx.view(-1, 1)).squeeze(1)

        rk = (F.relu(p("pair_other_i", c_i) - p("pair_self_i", c_i) + self.pair_eps)
              + F.relu(p("pair_other_j", c_j) - p("pair_self_j", c_j) + self.pair_eps)
              ).mean()

        total = ce + self.pair_rank_w * rk
        logs = {"pair_ce": float(ce.detach()), "pair_rank": float(rk.detach())}
        if "pair_gate_mean" in outputs:
            logs["pair_gate"] = float(outputs["pair_gate_mean"])
        return total, logs

    def forward(self, outputs, batch):
        # 先把「普通张量」和「dict」两种情况归一成一个 dict。
        # 以前这里是 if not dict: return self._one(...), {} —— 结果熵正则写在
        # dict 分支里，一旦模型返回普通张量（关掉全部辅助分支，或者 cutmix batch
        # 屏蔽了 CAM 分支），熵正则就被整段跳过、还不报错。
        out = outputs if isinstance(outputs, dict) else {"logits": outputs}

        loss = self.main_w * self._one(out["logits"], batch)
        logs = {}
        for key, w in self.aux_weights.items():
            if key in out and w > 0:
                aux = self._one(out[key], batch,
                                hard_label=key in self.hard_label_aux)
                loss = loss + w * aux
                logs[key] = float(aux.detach())

        # 裁剪一致性：对称 KL，两边都参与（不 detach），温度 T 把分布软化。
        if self.consistency_w > 0 and "logits_crop" in out:
            T = self.consistency_T
            m = F.log_softmax(out["logits"].float() / T, dim=1)
            c = F.log_softmax(out["logits_crop"].float() / T, dim=1)
            kl = ((m.exp() * (m - c)).sum(1) + (c.exp() * (c - m)).sum(1)).mean() * 0.5
            loss = loss + self.consistency_w * kl * T * T
            logs["consistency"] = float(kl.detach())
        # 部位视图的一致性：主分支 vs 每一路 GT 部位（head/wing/tail/body）。
        # 与 cam_crop 的 consistency 同一个算子、同一个温度，只是作用在额外的视图上。
        # 实测（batch4）裁剪分支的价值主要来自一致性（0.3 最好、0 显著更差），
        # 所以多路视图也应该有这条约束，而不是只有 CE。
        if self.part_view_cons_w > 0:
            T = self.consistency_T
            m = F.log_softmax(out["logits"].float() / T, dim=1)
            acc = []
            for k in ("logits_head", "logits_body", "logits_wing", "logits_tail"):
                if k in out:
                    c = F.log_softmax(out[k].float() / T, dim=1)
                    kl = ((m.exp() * (m - c)).sum(1)
                          + (c.exp() * (c - m)).sum(1)).mean() * 0.5
                    acc.append(kl)
            if acc:
                kl = torch.stack(acc).mean()
                loss = loss + self.part_view_cons_w * kl * T * T
                logs["part_consistency"] = float(kl.detach())
        if self.consistency2_w > 0 and "logits_crop2" in out:
            T = self.consistency_T
            m = F.log_softmax(out["logits"].float() / T, dim=1)
            c = F.log_softmax(out["logits_crop2"].float() / T, dim=1)
            kl = ((m.exp() * (m - c)).sum(1) + (c.exp() * (c - m)).sum(1)).mean() * 0.5
            loss = loss + self.consistency2_w * kl * T * T
            logs["consistency2"] = float(kl.detach())

        # 知识蒸馏：学生（主分支）追教师的软化分布
        if self.distill_w > 0 and "teacher_logits" in batch:
            T = self.distill_T
            pt = F.log_softmax(batch["teacher_logits"].float() / T, dim=1).detach()
            ps = F.log_softmax(out["logits"].float() / T, dim=1)
            kl = (pt.exp() * (pt - ps)).sum(1)          # 逐样本
            if self.distill_gate_label:
                y = batch["label"]
                if "soft_target" in batch:
                    y = batch["soft_target"].argmax(1)      # 混合 batch 取权重大的那类
                keep = (batch["teacher_logits"].float().argmax(1) == y.to(kl.device)).float()
                kl = kl * keep
                logs["distill_kept"] = float(keep.mean())
            if self.distill_min_conf > 0:
                # 教师在原始尺度上的最大概率（不是在温度 T 下）
                conf = F.softmax(batch["teacher_logits"].float(), dim=1).max(dim=1).values
                keep = (conf >= self.distill_min_conf).float()
                kl = kl * keep
                logs["distill_kept"] = float(keep.mean())
            loss = loss + self.distill_w * kl.mean() * T * T
            logs["distill"] = float(kl.mean().detach())

        if self.tl_reg_w > 0 and "tl_reg" in out:
            loss = loss + self.tl_reg_w * out["tl_reg"]
            logs["tl_reg"] = float(out["tl_reg"].detach())

        # 部件散度：让 K 个部件尽量看不同的地方（否则它们会退化成同一张 CAM）
        if self.part_cons_w > 0 and "part_cons" in out:
            loss = loss + self.part_cons_w * out["part_cons"]
            logs["part_cons"] = float(out["part_cons"].detach())

        if self.pa_div_w > 0 and "pa_div" in out:
            loss = loss + self.pa_div_w * out["pa_div"]
            logs["pa_div"] = float(out["pa_div"].detach())

        # 熵正则：减号 = 最大化熵，抑制「错得自信」
        if self.entropy_w > 0:
            lg = out["logits"].float()
            ent = -(F.softmax(lg, dim=1) * F.log_softmax(lg, dim=1)).sum(1).mean()
            loss = loss - self.entropy_w * ent
            logs["entropy"] = float(ent.detach())

        # 强视图：先给一个 CE（否则它只是被蒸馏、失去了抗坍缩压力），
        # 再加一条「主分支 → 强视图」的非对称 KL，教师侧 detach
        if "logits_strong" in out and (self.strong_ce > 0 or self.strong_kl > 0):
            if self.strong_ce > 0:
                l_strong = self._one(out["logits_strong"], batch)
                loss = loss + self.strong_ce * l_strong
                logs["strong_ce"] = float(l_strong.detach())
            if self.strong_kl > 0:
                T = self.strong_T
                pw = F.log_softmax(out["logits"].float() / T, dim=1).detach()
                ps = F.log_softmax(out["logits_strong"].float() / T, dim=1)
                kl = (pw.exp() * (pw - ps)).sum(1).mean()
                loss = loss + self.strong_kl * kl * T * T
                logs["strong_kl"] = float(kl.detach())

        # 成对判别：主分支照常算 CE，额外加一组成对损失（API-Net 的四路 CE + 排名）
        if self.pair_w > 0 and "pair_self_i" in out:
            pair_loss, pair_logs = self._pair_loss(out, batch)
            loss = loss + self.pair_w * pair_loss
            logs.update(pair_logs)
        return loss, logs


def build_criterion(cfg: dict) -> nn.Module:
    t = cfg["TRAIN"]
    return ClassificationCriterion(
        label_smoothing=t["label_smoothing"],
        aux_weights={
            "logits_drop":  t.get("cam_drop", 0.0),
            "logits_crop":  t.get("cam_crop", 0.0),
            "logits_quad":  t.get("cam_quad", 0.0),
            "logits_parts": t.get("attn_weight", 0.0),
            # 部件注意力的「每个部件都要能分类」辅助损失（与旧 attn_aux 的
            # logits_parts 分开命名，避免两套模块互相串权重）
            "logits_pa":    t.get("pa_aux_w", 0.0),
            # 逐 token 辅助 CE（TokenLearner replace 出口）：每个部件都要能单独分类
            "logits_tl":    t.get("tl_aux_w", 0.0),
            # 训练期特权裁剪的额外部位视图（wing/tail/body），各自接硬标签 CE
            "logits_head":  t.get("cam_part_w", 0.0),
            "logits_body":  t.get("cam_part_w", 0.0),
            "logits_wing":  t.get("cam_part_w", 0.0),
            "logits_tail":  t.get("cam_part_w", 0.0),
            # 双路融合（整图 + CAM 放大）的总损失权重
            "logits_fused": t.get("cam_fuse_w", 0.0),
            "logits_hyp1": t.get("hyp_aux_w", 0.0),
            "logits_partaux": t.get("pta_aux_w", 0.0),
            "logits_mge": t.get("mge_aux_w", 0.0),
            "logits_region": t.get("rr_region_w", 0.0),
            "logits_hyp2": t.get("hyp_aux_w", 0.0),
            "logits_ds": t.get("deep_sup_w", 0.0),
            "logits_pmg1": t.get("pmg_stage_w", 0.0),
            "logits_pmg2": t.get("pmg_stage_w", 0.0),
            "logits_pmg3": t.get("pmg_stage_w", 0.0),
        },
        hard_label_aux=tuple(
            k for k, on in (("logits_crop", t.get("cam_crop_hard_label", False)),
                            ("logits_head", bool(t.get("cam_part_views"))),
                            ("logits_body", bool(t.get("cam_part_views"))),
                            ("logits_wing", bool(t.get("cam_part_views"))),
                            ("logits_tail", bool(t.get("cam_part_views"))))
            if on
        ),
        consistency_w=t.get("cam_consistency", 0.0),
        consistency_T=t.get("cam_consistency_T", 2.0),
        consistency2_w=t.get("cam_consistency2", 0.0),
        pair_w=t.get("pair_w", 0.0),
        pair_eps=t.get("pair_eps", 0.05),
        pair_rank_w=t.get("pair_rank_w", 1.0),
        hard_neg_k=t.get("hard_neg_k", 0),
        entropy_w=t.get("entropy_w", 0.0),
        main_w=t.get("main_w", 1.0),
        distill_w=t.get("distill_w", 0.0),
        distill_T=t.get("distill_T", 2.0),
        distill_min_conf=t.get("distill_min_conf", 0.0),
        distill_gate_label=bool(t.get("distill_gate_label", False)),
        tl_reg_w=t.get("tl_reg_w", 0.0),
        pa_div_w=t.get("pa_div_w", 0.0),
        part_cons_w=t.get("pta_cons_w", 0.0),
        part_view_cons_w=t.get("cam_part_cons", 0.0),
        strong_ce=t.get("strong_ce", 0.0),
        strong_kl=t.get("strong_kl", 0.0),
        strong_T=t.get("strong_T", 2.0),
    )


def _stage_of(param_name: str) -> str:
    """把 backbone 参数名映射到 stage：conv1/bn1 -> stem，其余按 layer1~4。"""
    # GeM 池化的 p 是随机初始化的新参数，不该跟着 stem 拿 0.01 倍学习率，
    # 单独成桶、回落到 backbone_lr_scale
    if param_name.startswith("global_pool"):
        return "pool"
    # 插在 layer3/layer4 之间的全局自注意力（model/global_ctx.py）。它是从零
    # 初始化的新参数，不能跟着 stem 拿 0.01 倍学习率 —— 那个倍率下零初始化
    # 的输出层永远长不起来。回落值是 backbone_lr_scale。
    if param_name.startswith("global_ctx"):
        return "global_ctx"
    if param_name.startswith("part_attn"):
        return "part_attn"
    # CLIP 版 ResNet 把注意力池化头放在 backbone.head（AttentionPool2d，12.7M）。
    # 它虽然叫 head，但属于 backbone，且是预训练过的；按 stem 的 0.01 倍学习率
    # 更新几乎等于冻结。给一个独立桶，回落 backbone_lr_scale。
    if param_name.startswith("head."):
        return "head_pool"
    for s in ("layer1", "layer2", "layer3", "layer4"):
        if param_name.startswith(s):
            return s
    # CLIP 版 ResNet（timm ByobNet）把四个 stage 放在 stages.0 ~ stages.3，
    # 名字和 torchvision 的 layer1~4 对不上。不映射的话整个主干都会落到 stem
    # 的 0.01 倍学习率上，和 R50 的配方就没法比了。
    if param_name.startswith("stages."):
        idx = param_name.split(".")[1]
        return {"0": "layer1", "1": "layer2",
                "2": "layer3", "3": "layer4"}.get(idx, "stem")
    return "stem"


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    tcfg = cfg["TRAIN"]
    base_lr = tcfg["lr"]
    wd = tcfg["weight_decay"]
    # 判别式学习率：预训练 backbone 用 base_lr × scale，新初始化的头用 base_lr。
    # 预训练权重不该被和随机初始化的头一样大的 lr 扰动 —— PMG (ECCV'20) 等微调
    # 配方给预训练层 1/10 的 lr（新层 0.002 / 预训练层 0.0002）。
    scale = tcfg.get("backbone_lr_scale", 1.0)

    # 只保留真正指定了的层（配置里默认是 null，表示"没启用"）
    layers_cfg = {k: v for k, v in (tcfg.get("backbone_lr_layers") or {}).items()
                  if v is not None}

    def split(module, m_scale: float, tag: str) -> list[dict]:
        """按「是否套 weight decay」把模块一分为二。

        bias 和归一化层的权重（都是一维参数）不套 wd —— 对它们衰减会持续把参数往 0 拉。
        "part" 用来事后识别哪一组是 backbone（渐进解冻要单独抬它的 lr）。
        """
        decay, no_decay = [], []
        for param in module.parameters():
            if not param.requires_grad:
                continue
            # ⚠️ 0 维标量（GeM.p、CosineHead.scale）也必须免 wd：
            # 只判 ndim==1 会让它们落进 wd=0.05 的组，被持续往 0 拉
            (no_decay if param.ndim <= 1 else decay).append(param)
        return [
            {"params": decay, "lr": base_lr * m_scale, "weight_decay": wd, "part": tag},
            {"params": no_decay, "lr": base_lr * m_scale, "weight_decay": 0.0, "part": tag},
        ]

    groups = []
    if hasattr(model, "backbone") and hasattr(model, "head"):
        if layers_cfg:
            # 分层学习率：按 stage 给不同倍率，低层小、高层大。
            # 动机（实测）：SWSL 权重的 backbone 只要 lr 从 1e-4 抬到 1.4e-4，
            # val 就掉 1 个点，到 1.8e-4 掉 5.6 —— 但 a1 用 1e-3 都没事。
            # 说明不能拿一个统一倍率对待整个 backbone。
            buckets: dict[str, list] = {}
            scales: dict[str, float] = {}
            for pname, param in model.backbone.named_parameters():
                if not param.requires_grad:
                    continue
                st = _stage_of(pname)
                buckets.setdefault(st, []).append(param)
                scales[st] = float(layers_cfg.get(st, scale))
            for st, params in buckets.items():
                decay = [p for p in params if p.ndim != 1]
                no_decay = [p for p in params if p.ndim == 1]
                groups.append({"params": decay, "lr": base_lr * scales[st],
                               "weight_decay": wd, "part": "backbone", "stage": st})
                groups.append({"params": no_decay, "lr": base_lr * scales[st],
                               "weight_decay": 0.0, "part": "backbone", "stage": st})
        else:
            groups += split(model.backbone, scale, "backbone")
        groups += split(model.head, 1.0, "head")
        # 这些辅助模块既不在 backbone 也不在 head 里。不单独列出来的话
        # 根本不会被更新 —— 训练会安静跑完，模块保持随机初始化。
        # 它们都是随机初始化的新参数，所以给头部的学习率（1.0）。
        for attr in ("fpn", "attn_aux", "pair_aux", "pmg", "deep_sup", "attn_pool", "se_gate",
                     "tokenlearner", "part_fusion", "part_head", "hyp_comb", "mge",
                     "region_router"):
            aux = getattr(model, attr, None)
            if aux is not None:
                m_scale = 1.0
                m_wd = None                      # None = 跟随全局 weight_decay
                if attr == "attn_pool":
                    # attn 模块是从零初始化的新参数，跟头共用 1e-3 未必合适；
                    # 允许 TRAIN.attn_lr 单独指定绝对学习率
                    _al = tcfg.get("attn_lr")
                    if _al:
                        m_scale = float(_al) / base_lr
                    _wd = tcfg.get("attn_wd")
                    if _wd is not None:
                        m_wd = float(_wd)
                g = split(aux, m_scale, attr)
                if m_wd is not None:
                    for gg in g:
                        gg["weight_decay"] = m_wd
                groups += g
    else:
        groups = split(model, 1.0, "all")

    # 双路融合的标量 β：它挂在 Classifier 上，既不在 backbone 也不在 head。
    # 不给它单独一组的话，下面的兜底检查会直接报错（这正是我们想要的）。
    for bname in ("cam_beta", "cam_beta2", "hyp_beta1", "hyp_beta2", "pr_cam_beta", "pa_beta"):
        beta = getattr(model, bname, None)
        if beta is not None:
            groups.append({"params": [beta], "lr": base_lr,
                           "weight_decay": 0.0, "part": bname})

    # GlobalContext（插在 backbone 里的全局自注意力）单独调 lr / weight decay。
    # 它从零初始化，输出层又很小；用全局 wd=0.05 会把刚长出来的输出权持续往 0 拉，
    # 而默认的 lr 是 backbone_lr_scale=1.0（比 layer3/4 大 20 倍）。
    gc_lr = tcfg.get("gc_lr")
    gc_wd = tcfg.get("gc_wd")
    pa_lr = tcfg.get("pa_lr")
    pa_wd = tcfg.get("pa_wd")
    if gc_lr is not None or gc_wd is not None or pa_lr is not None or pa_wd is not None:
        for g in groups:
            st = g.get("stage")
            if st == "global_ctx":
                if gc_lr is not None:
                    g["lr"] = float(gc_lr)
                if gc_wd is not None:
                    g["weight_decay"] = float(gc_wd)
            elif st == "part_attn":
                if pa_lr is not None:
                    g["lr"] = float(pa_lr)
                if pa_wd is not None:
                    g["weight_decay"] = float(pa_wd)

    # 兜底：任何可训练参数没进优化器就直接报错。
    # 这个坑已经踩过两次（第一次是 attn_aux / pair_aux，第二次是 pmg / deep_sup / attn_pool），
    # 两次都是"训练照常跑、指标照常出，只是那个模块一直没被更新"。
    covered = {id(p) for g in groups for p in g["params"]}
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and id(p) not in covered]
    if missing:
        raise RuntimeError(
            f"有 {len(missing)} 个可训练参数没有进优化器，它们会一直保持初始化值："
            f"{missing[:6]}")

    name = tcfg["optimizer"].lower()
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=base_lr)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=base_lr, momentum=tcfg["momentum"])
    raise KeyError(f"未知的优化器: {name!r}，可选 adamw | sgd")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict):
    """按 epoch 推进的调度器。返回 None 表示恒定学习率。

    fit() 里已经处理了 None 的情况，所以调用方不用判断。
    """
    tcfg = cfg["TRAIN"]
    name = (tcfg.get("scheduler") or "none").lower()
    if name == "none":
        return None

    epochs = tcfg["epochs"]
    warmup_epochs = tcfg.get("warmup_epochs", 0)
    # LambdaLR 的返回值是 lr 的**倍率**，所以要换算成相对初始 lr 的比例
    min_lr_ratio = tcfg.get("min_lr", 0.0) / tcfg["lr"] if tcfg["lr"] > 0 else 0.0

    if name == "cosine":
        # lr_decay_frac < 1 时，cosine 在训练结束之前就降到 min_lr，剩下的轮次
        # 维持 min_lr 继续精调。默认 1.0 = 和以前完全一样（整条曲线走满 epochs）。
        # 例：epochs=50, lr_decay_frac=0.6 -> 第 30 轮就降到 min_lr。
        decay_frac = float(tcfg.get("lr_decay_frac", 1.0) or 1.0)
        decay_epochs = max(int(round((epochs - warmup_epochs) * decay_frac)), 1)

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / max(warmup_epochs, 1)
            progress = min((epoch - warmup_epochs) / decay_epochs, 1.0)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(epochs // 3, 1), gamma=0.1
        )

    raise KeyError(f"未知的调度器: {name!r}，可选 cosine | step | none")
