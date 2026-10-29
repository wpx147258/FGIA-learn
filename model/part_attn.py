"""部件提取 + 部件间自注意力（综述 [70]–[76] 那一族，用自注意力替换 LSTM/图传播）。

文献线索（Wei et al. TPAMI 2022 的 Figure 7 / Table 2）：
  [76] Xiao et al. CVPR'15 用深度滤波器的谱聚类得到「部件检测器」组；
  [70] Zhang et al. TIP'16 用空间金字塔提议 + 聚类得到部件原型；
  [72] He & Peng AAAI'17 用空间约束做部件选择；
  [73] Ge et al. CVPR'19 自底向上做互补部件；
  [74] Wang et al. AAAI'20（GCL）把部件当作图节点、用「十字交叉图传播」让部件互看；
  [75] Liu et al. AAAI'20 做区域注意力的过滤与蒸馏；
  [71] Lam et al. CVPR'17（HSnet）用 LSTM 在部件之间串行搜索。

它们的共同点是「先拿到部件，再让部件之间交换信息」；差别只在融合器（LSTM / 图 / 蒸馏）。
我们用**自注意力**做这个融合器：K 个部件描述子作为 K 个 token，互相看一层，
再汇总回特征。这与我们之前做过的两件事都不同：
  * AttnPool：token 是 14×14 的空间格子，不是部件；
  * TokenLearner：有部件（软掩码），但**部件之间不说话**，只是取平均。

约束与其它模块一致：残差输出零初始化，训练第 0 步与基线逐位等价；
输出加在特征图上（GAP 之前），因此 eval、CAM 裁剪分支、y=None 早退路径
全部走同一条路径（用 layer4 的 forward hook 实现，不依赖 Classifier 的分支顺序）。

结构（448 输入，layer4 输出 [B,2048,14,14]，K=16）：
    x [B,C,h,w]
      -> 1×1 检测器           [B,K,h,w] -> softmax over (h,w) => 每个部件的空间分布
      -> 加权池化             [B,K,C]    （第 k 行 = 第 k 个部件的描述子）
      -> 线性降到 d + 部件身份嵌入
      -> 1 层 pre-LN 自注意力  [B,K,d]   （部件之间互相看）
      -> 平均 + LayerNorm      [B,d]
      -> 线性回 C（零初始化）   [B,C] -> 加到特征图的每个位置
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.global_ctx import _sincos_2d


class PartAttention(nn.Module):
    def __init__(self, channels: int = 2048, num_parts: int = 16, dim: int = 256,
                 depth: int = 1, heads: int = 4, mlp_ratio: float = 2.0,
                 dropout: float = 0.0, tau: float = 1.0, margin: float = 0.02,
                 compete: bool = False, norm_spatial: bool = False,
                 num_classes: int = 0, grouped: bool = False,
                 res_scale: float = 0.0, no_residual: bool = False,
                 detach_feat: bool = False, perm_path: str = "", kernel: int = 1,
                 js_w: float = 0.0, js_target: float = 0.35,
                 mul: bool = False, mul_center: bool = True, mul_gate: float = -10.0,
                 mask_blur: bool = False, readout: str = "mean",
                 std_correction: int = 0, random_masks: bool = False):
        super().__init__()
        self.num_parts = int(num_parts)
        # 竞争式划分：每个空间位置在 K 个部件之间做 softmax（而不是每个部件
        # 独立在空间上归一化）。这是「把部件推开」的结构性做法：一个格子只能
        # 属于一个部件，配合负载均衡项，K 个部件的支撑集必然不同。
        # 上一版每个部件各自 softmax，在平坦响应上会退化成同一张近均匀图。
        self.compete = bool(compete)
        # 先做空间归一化（减去该图的空间均值、除以空间标准差）再竞争：
        # 否则响应图的绝对高低由通道均值决定，所有部件会一起偏到同一处。
        self.norm_spatial = bool(norm_spatial)
        self._last_balance = None
        self.dim = int(dim)
        self.tau = float(tau)
        # MA-CNN 散度项的 margin（原文 mrg = 0.02）
        self.margin = float(margin)
        # 部件检测器：每个通道图是一个"这个部件在哪里"的响应。
        # grouped=True 是 MA-CNN 的关键做法——把通道硬分成 K 组，第 k 个检测器
        # 只能看第 k 组的通道（Conv2d groups=K 就是按通道切分）。这样四个部件
        # **从一开始就用不相交的通道子空间**，从结构上破坏了「四个检测器退化成
        # 同一张图」的对称性；散度项只负责维持分离，不再孤军奋战。
        self.grouped = bool(grouped)
        # 残差幅度控制。实测 pa4 在部署权重上 ||res||/||GAP||=0.71、cos(res,GAP)=-0.69，
        # 等于用一条反相关的通道把主特征削掉 28%；res_scale>0 时把残差范数固定成
        # GAP 范数的 res_scale 倍（缩放系数 detach，不参与梯度）。
        self.res_scale = float(res_scale)
        # 乘性空间残差：output = output * (1 + γ · Σ_k m_k)，γ = sigmoid(mul_gate)。
        # 与加性残差的本质区别：这里改的是**每个位置的增益**，GAP 之后位置权重真的变了；
        # 加性版本把 [B,C] 广播到所有位置，等价于逐通道常数偏移，表达不了空间重加权。
        # mask_blur：掩码先做 3×3 平均池化再用。原因（2026-10-29 诊断）：
        # 掩码是「铺满全图的碎斑」——熵 4.45~4.96/5.28、平均 13 个连通块、邻居相关只有 0.27。
        # 根因是 1×1 检测器 + 逐格标准化，没有任何空间平滑先验；而输出只是全图格子的
        # 加权平均，模型没有动力让掩码成片。3×3 平滑是把它变回「区域」的最小改动。
        self.mask_blur = bool(mask_blur)
        # 部件 token 的读出方式：
        #   mean —— 直接对 K 个部件 token 取平均（历史默认）；
        #   cls  —— 在最前面拼一个全局 token（初值=部件 token 的均值），过自注意力后读它。
        # 为什么要有 cls：AttnPool 上 mean→cls 把 88.18 拉到 88.99（−0.88 → −0.07），
        # 说明「读出方式」在这类模块上不是小事；而 PartAttention 一直只有 mean 读出，
        # 这是一处此前没有意识到的对照缺失。
        if readout not in ("mean", "cls"):
            raise ValueError(f"pa_readout 只能是 mean | cls，收到 {readout!r}")
        self.readout = readout
        # norm_spatial 里 std 的分母自由度：0 → N（当前默认），1 → N−1（历史默认）。
        # 2026-10-29 03:25 的审计把它从 torch 默认的 1 改成了 0，等于把检测器分数整体
        # 缩放 0.26%。这个开关是为了**精确复现历史 run**（原 pa4_js 用的是 correction=1，
        # 即 N−1=195），不是调参项。
        self.std_correction = int(std_correction)
        # 对照：用**固定的随机掩码**替换学到的掩码（不参与梯度）。
        # 用途：如果「随机掩码」与「学到的掩码」成绩相同，说明空间结构根本没起作用，
        # 部件线的收益（如果有）只来自通道分组 / 逐部件 CE / 额外参数。
        self.random_masks = bool(random_masks)
        self.register_buffer("_rand_m", None, persistent=False)
        self.mul = bool(mul)
        # mul_center：先把每张掩码减去自己的空间均值，使 Σ_k m 的均值为 0、
        # 增益 1+Σm 的均值为 1（初始平均不改尺度）。不居中时 K 张 sigmoid 掩码
        # 的均值约 K/2，整幅特征会被放大 (1+K/2) 倍，等于把预训练尺度打乱。
        self.mul_center = bool(mul_center)
        # ⚠️ 只有开 mul 时才创建这两个参数：否则旧 checkpoint（没有 mul_gate）
        # 会因为 state_dict 缺键而加载失败（实测踩过）。
        self.mul_gate = nn.Parameter(torch.tensor([float(mul_gate)])) if self.mul else None
        # ⚠️ 与 compete 的交互（实测发现）：compete=True 时掩码在每个位置对 K 求和恒为 1，
        # 因此「各掩码减去空间均值后再求和」恒等于 0，乘性增益会退化成恒等（实测 std 5e-8）。
        # 解决办法：每个部件配一个可学习权重 c_k（随机小初始化），gain = 1 + γ·Σ c_k·m̃_k。
        self.mul_pw = None
        if self.mul and self.compete:
            self.mul_pw = nn.Parameter(torch.randn(self.num_parts) * 0.1)
        # JS 正则（与 TokenLearner 同一套标尺）：N=196 时 JS=0.693 表示支撑完全不重叠，
        # 0.35 表示最多重合一半，0 表示同一个分布。hinge 形式只惩罚「太像」。
        # 为什么不用内积：w 是概率分布，两张「都偏均匀」的图内积天然很小（1/N 量级），
        # 会把「同一张均匀图」误判成「不重叠」；JS 不会。
        self.js_w = float(js_w)
        self.js_target = float(js_target)
        # 完全去掉残差：部件证据改走 logit 级融合（见 Classifier 的 pa_logit_fuse）
        self.no_residual = bool(no_residual)
        # 部件分支与主干解耦：检测器/池化只看 detach 后的特征，逐部件 CE 不再回传
        # 到 backbone。这样主分支与基线逐位同分布（只受 CAM 分支影响），
        # 否决票是完全叠加在上面的零成本机制。
        self.detach_feat = bool(detach_feat)
        # 通道分组的顺序：连续切分是索引任意的（torch 的通道顺序没有语义），
        # 用训练集统计的通道相关性聚类得到的 perm 会把相关通道放在同一组。
        if perm_path:
            import numpy as np
            perm = torch.as_tensor(np.load(perm_path), dtype=torch.long)
            self.register_buffer("perm", perm, persistent=False)
        else:
            self.perm = None
        if self.grouped:
            if channels % self.num_parts != 0:
                raise ValueError(f"grouped 需要 channels({channels}) 能被 num_parts({self.num_parts}) 整除")
            kk = int(kernel); pad = kk // 2
            self.det = nn.Conv2d(channels, self.num_parts, kernel_size=kk, padding=pad,
                                 groups=self.num_parts)
            self.group_ch = channels // self.num_parts
        else:
            kk = int(kernel); pad = kk // 2
            if kk < 1 or kk % 2 == 0:
                raise ValueError(f"pa_kernel 必须是正奇数（偶数核会改变空间尺寸），收到 {kk}")
            self.det = nn.Conv2d(channels, self.num_parts, kernel_size=kk, padding=pad)
            self.group_ch = channels
        # 部件身份嵌入：检测器是有编号的，注意力需要知道"我是第几个部件"
        self.emb = nn.Parameter(torch.zeros(1, self.num_parts, dim))
        nn.init.trunc_normal_(self.emb, std=0.02)
        self.proj = nn.Linear(self.group_ch, dim, bias=False)
        if depth > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        else:
            # 控制臂：同样的参数预算，但**不做部件间混合**（逐点 MLP）
            self.blocks = nn.Sequential(
                nn.Linear(dim, int(dim * mlp_ratio * 2)), nn.GELU(),
                nn.Linear(int(dim * mlp_ratio * 2), dim))
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # 逐部件分类头（MA-CNN 的 part-CNN 的轻量替代）：只在开了部件 CE 时用。
        # 用独立的小头而不是共享主头，是为了不让部件损失直接改写主分类器的权重。
        self.part_cls = nn.Linear(self.group_ch, int(num_classes)) if num_classes else None
        self._last_part_logits = None
        # 诊断：部件分布的熵（越小越像"每个部件只看一处"）与部件间重叠（越大越塌缩）
        self._last_entropy = 0.0
        self._last_overlap = 0.0
        self._last_ratio = 0.0
        # 给分类头用的部件描述子（[B,K,C]）和可微的散度项，训练时由 Classifier 取走
        self._last_z: torch.Tensor | None = None
        self._last_div: torch.Tensor | None = None

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if self.detach_feat:
            feat = feat.detach()
        if self.perm is not None:
            feat = feat[:, self.perm]
        b, c, h, w = feat.shape
        # MA-CNN Eq.4：部件掩码用 sigmoid 得到 [0,1] 的响应，而不是 softmax。
        # 这一点很关键：散度项是在 [0,1] 的掩码上做的（margin=0.02），
        # 而 softmax 后的值只有 1/N（约 0.005），两者尺度差两个数量级，
        # 直接套 hinge 会永远落在 margin 下方、等于没加正则。
        if self.random_masks:
            if self._rand_m is None or self._rand_m.shape[-2:] != (h, w):
                g = torch.Generator(device=feat.device).manual_seed(1234)
                m0 = torch.rand(1, self.num_parts, h, w, generator=g, device=feat.device)
                self._rand_m = (m0 / m0.sum(dim=(2, 3), keepdim=True) * (h * w) * 0.5).clamp(1e-3, 1.0)
            m = self._rand_m.expand(b, -1, -1, -1)        # 固定、跨样本相同、不学习
            raw = m.flatten(2)
        else:
            raw = self.det(feat).flatten(2)               # [B,K,N]
        if self.norm_spatial:
            raw = (raw - raw.mean(dim=-1, keepdim=True)) / raw.std(dim=-1, keepdim=True, correction=self.std_correction).clamp_min(1e-5)
        if self.compete:
            logits = raw / self.tau
            A = logits.softmax(dim=1)                    # 每个位置在 K 个部件间竞争
            n_pos = A.shape[-1]
            mass = A.sum(-1) / n_pos                     # [B,K] 每个部件占的份额
            # 负载均衡：份额应当接近 1/K，否则会塌成「一个部件吃掉全图」
            self._last_balance = (self.num_parts * ((mass - 1.0 / self.num_parts) ** 2).sum(-1)).mean()
            m = A / A.sum(-1, keepdim=True).clamp_min(1e-6)
        else:
            m = torch.sigmoid(raw)                       # 旧路径：每个部件独立 [0,1] 掩码
        if self.mask_blur:
            # ⚠️ 2026-10-30 修复：这条分支以前只写在 part_masks() 里（只有 mul 路径和探针会调用），
            # 主前向在这里内联重算掩码，于是 pa_mask_blur=true 对训练**完全无效**——
            # 实测 all_sig 与 all_sig_blur 五个权重 test 逐位相同、只差 EXP_NAME。
            m = F.avg_pool2d(m.view(b, self.num_parts, h, w),
                             kernel_size=3, stride=1, padding=1).view(b, self.num_parts, h * w)
        a = m / m.sum(dim=-1, keepdim=True).clamp_min(1e-6)   # 归一化后用于池化
        if self.grouped:
            # 每个部件只用自己那组通道做加权池化（MA-CNN Eq.5 的通道分组版本）
            xs = feat.view(b, self.num_parts, self.group_ch, h, w)
            mk = m.view(b, self.num_parts, 1, h, w)
            z = (xs * mk).sum(dim=(3, 4)) / mk.sum(dim=(3, 4)).clamp_min(1e-6)
        else:
            f = feat.flatten(2)                              # [B,C,N]
            z = torch.einsum("bkn,bcn->bkc", a, f)           # [B,K,C]
        t = self.proj(z) + self.emb                      # [B,K,d]
        if self.readout == "cls":
            # 全局 token 初值取部件 token 的均值（与 AttnPool / TokenLearner 的 CLS 一致）
            g = t.mean(dim=1, keepdim=True)              # [B,1,d]
            tt = self.blocks(torch.cat([g, t], dim=1))   # [B,K+1,d]
            pooled = self.norm(tt)[:, 0]                 # [B,d]
        else:
            t = self.blocks(t)
            pooled = self.norm(t).mean(dim=1)            # [B,d]
        y = self.out(pooled)                             # [B,C]

        # ---- MA-CNN 的两项正则（L_cng = Dis + λ·Div，λ=2）----
        k = self.num_parts
        if k > 1:
            eye = torch.eye(k, device=m.device)
            others = m.unsqueeze(1) * (1.0 - eye)[None, :, :, None]   # [B,K,K,N]
            max_other = others.max(dim=2).values                     # [B,K,N]
            # Div：不同部件的掩码在同一位置不要都高（hinge，margin 0.02）
            # 原文是对位置求和；我们取平均，因为原文的掩码是稀疏的（每个部件只覆盖
            # 一小块），而我们未训练的检测器给的是接近 0.5 的稠密图，直接求和会到
            # 几百的量级、把主损失压掉。
            div = (m * (max_other - self.margin).clamp_min(0.0)).sum(-1).mean() / (h * w)
            # Dis：每个部件的掩码要紧凑地围绕自己的质心
            gy = ((torch.arange(h, device=m.device, dtype=m.dtype) + 0.5) / h)
            gx = ((torch.arange(w, device=m.device, dtype=m.dtype) + 0.5) / w)
            yy = gy.view(1, 1, h, 1).expand(b, k, h, w).reshape(b, k, -1)
            xx = gx.view(1, 1, 1, w).expand(b, k, h, w).reshape(b, k, -1)
            cy = (a * yy).sum(-1, keepdim=True)
            cx = (a * xx).sum(-1, keepdim=True)
            dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
            dis = (m * dist2).sum(-1).mean() / (h * w)
            self._last_div = (self._last_balance if self.compete
                              else dis + 2.0 * div)
            if self.js_w > 0 and k > 1:
                # 两两 JS：p_k 与 p_m 在位置维 N 上比较
                mm = (a[:, :, None, :] + a[:, None, :, :]) / 2.0   # [B,K,K,N]
                la = a.clamp_min(1e-9).log()[:, :, None, :]        # [B,K,1,N]
                lm = mm.clamp_min(1e-9).log()
                kl1 = (a[:, :, None, :] * (la - lm)).sum(-1)       # [B,K,K]
                js = 0.5 * (kl1 + kl1.transpose(1, 2))
                off_ = 1.0 - torch.eye(k, device=a.device)
                js_off = (js * off_).sum(dim=(1, 2)) / (k * (k - 1))
                self._last_div = self._last_div + self.js_w * (self.js_target - js_off).clamp_min(0).mean()
        else:
            self._last_div = torch.zeros((), device=m.device)

        if self.part_cls is not None:
            self._last_part_logits = self.part_cls(z)

        # 训练时要用的两样东西：部件描述子（给分类头做逐部件 CE）和可微正则项
        self._last_z = z
        with torch.no_grad():
            ent = -(a * a.clamp_min(1e-9).log()).sum(-1)
            self._last_entropy = float(ent.mean())
            a_flat = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            sim = torch.einsum("bkn,bmn->bkm", a_flat, a_flat)
            off = ((sim * (1.0 - torch.eye(k, device=a.device))).sum(dim=(1, 2))
                   / max(k * (k - 1), 1))
            self._last_overlap = float(off.mean())
            g = feat.detach().flatten(2).norm(dim=2).mean()
            self._last_ratio = float(y.detach().norm().mean() / g.clamp(min=1e-12))
        return y

    def part_masks(self, feat: torch.Tensor) -> torch.Tensor:
        """返回 [B,K,h,w] 的部件掩码；口径与 forward 内部一致（sigmoid 或竞争 softmax）。"""
        if self.perm is not None:
            feat = feat[:, self.perm]
        raw = self.det(feat)
        b, k, h, w = raw.shape
        raw = raw.flatten(2)
        if self.norm_spatial:
            raw = (raw - raw.mean(dim=-1, keepdim=True)) / raw.std(dim=-1, keepdim=True, correction=self.std_correction).clamp_min(1e-5)
        if self.compete:
            m = (raw / self.tau).softmax(dim=1)
        else:
            m = torch.sigmoid(raw)
        m = m.view(b, k, h, w)
        if self.mask_blur:
            # 3×3 平均（步长 1、padding 1）后再归一化：把碎斑连成片。
            # 注意 compete 下 sum_k m = 1，池化后仍然成立（平均是线性算子）。
            m = F.avg_pool2d(m, kernel_size=3, stride=1, padding=1)
        return m

    def hook(self, module, inputs, output):
        """layer4 的 forward hook：GAP 之前把部件注意力的结果加进去。

        GAP 是线性的，所以"加在特征图上"等价于"加在池化后的向量上"；
        选前者是为了让 CAM 裁剪分支、y=None 早退路径、PMG、BN 重估全部自动经过它。
        """
        if self.mul:
            # 乘性空间残差：用掩码给每个位置一个增益，GAP 看到的是重加权后的特征图。
            # 初始 γ≈0 → 增益恒为 1 → 与基线逐位等价。
            m = self.part_masks(output)                          # [B,K,h,w]
            if self.mul_center:
                m = m - m.mean(dim=(2, 3), keepdim=True)         # 均值 0 → 增益均值 1
            if self.mul_pw is not None:
                # compete 模式下各掩码和为 1，必须用不同的 c_k 才能得到非退化的空间增益
                w = (m * self.mul_pw.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
            else:
                w = m.sum(dim=1, keepdim=True)                   # [B,1,h,w]
            g = torch.sigmoid(self.mul_gate)                     # 初始≈4.5e-5
            self._last_res = (g * w).detach()
            return output * (1.0 + g * w)
        y = self.forward(output)
        # 诊断用：bypass=True 时完全不加残差，用来量「部件分支到底改了多少」
        if getattr(self, "bypass", False):
            return output
        if self.no_residual:
            return output
        if self.res_scale > 0:
            # y 是 [B,C]（池化后的残差向量），不是 4D 特征图。
            # 用 s = r·g/(‖y‖+g)：初始 ‖y‖≈0 时 s≈r（有界），事后趋近 r·g/‖y‖。
            # 旧写法 r·g/‖y‖ 在零初始化时 s≈r·g/1e-6，反传放大约 4e5 倍（审计发现）。
            g = output.mean(dim=(2, 3)).norm(dim=1)                 # [B]
            yn = y.norm(dim=1)                                      # [B]
            s = (self.res_scale * g / (yn + g).clamp_min(1e-6)).detach()
            y = y * s[:, None]
        self._last_res = y.detach()
        # ⚠️ 注意这里的广播：y 是 [B,C]，加在 [B,C,1,1] 上 = **每个空间位置加同一个通道向量**。
        # 由于 GAP 线性，送到分类器的就是 pooled = GAP + y —— 一个逐通道的加性偏移。
        # 也就是说默认残差出口**无法改变空间权重**，部件掩码只影响 y 的取值，
        # 不影响池化时各位置的权重。要真正做空间重加权，必须乘在特征图上或替换池化。
        return output + y[:, :, None, None]


def build_part_attention(backbone: nn.Module, mcfg: dict,
                         num_classes: int = 0) -> PartAttention | None:
    if not mcfg.get("part_attn", False):
        return None
    if not hasattr(backbone, "layer4"):
        raise ValueError("MODEL.part_attn 目前只支持 ResNet 系列骨干")
    if getattr(backbone, "part_attn", None) is not None:
        raise RuntimeError("part_attn 已经注册过了，不要重复调用")
    # 检测器挂在哪一层：layer4（14×14，语义强但空间同质）或 layer3（28×28，
    # 分辨率够、区域探针显示还有定位能力）。默认 layer4，保持旧实验口径。
    src = str(mcfg.get("pa_from", "layer4"))
    if src not in ("layer3", "layer4"):
        raise ValueError(f"MODEL.pa_from 只能是 layer3 / layer4，收到 {src!r}")
    target = getattr(backbone, src)
    ref = target[-1]
    ch = getattr(getattr(ref, "conv3", None), "out_channels", None)
    if ch is None:
        raise ValueError(f"{src} 的最后一个 block 没有 conv3")
    if mcfg.get("pa_multiscale", False):
        return build_multiscale_parts(backbone, mcfg, num_classes)
    if mcfg.get("pa_slots", False):
        if mcfg.get("pa_logit_fuse", False):
            raise ValueError("SlotParts 没有通道组描述子（_last_z），不能和 pa_logit_fuse 同开；"
                             "请用 pa_no_residual=False（残差出口）或改用 PartAttention")
        sl = SlotParts(
            channels=ch,
            num_parts=int(mcfg.get("pa_parts", 4)),
            dim=int(mcfg.get("pa_dim", 256)),
            iters=int(mcfg.get("pa_slot_iters", 3)),
            mlp_ratio=float(mcfg.get("pa_mlp", 2.0)),
            dropout=float(mcfg.get("pa_dropout", 0.0)),
            num_classes=int(num_classes),
        )
        # 选项必须真的传下去（旧版静默忽略 detach / no_residual / res_scale）
        sl.detach_feat = bool(mcfg.get("pa_detach", False))
        sl.no_residual = bool(mcfg.get("pa_no_residual", False))
        sl.res_scale = float(mcfg.get("pa_res_scale", 0.0))
        try:
            p = next(backbone.parameters())
            sl.to(device=p.device, dtype=p.dtype)
        except StopIteration:
            pass
        backbone.add_module("part_attn", sl)
        target.register_forward_hook(sl.hook)
        return sl
    pa = PartAttention(
        channels=ch,
        num_parts=int(mcfg.get("pa_parts", 16)),
        dim=int(mcfg.get("pa_dim", 256)),
        depth=int(mcfg.get("pa_depth", 1)),
        heads=int(mcfg.get("pa_heads", 4)),
        mlp_ratio=float(mcfg.get("pa_mlp", 2.0)),
        dropout=float(mcfg.get("pa_dropout", 0.0)),
        tau=float(mcfg.get("pa_tau", 1.0)),
        margin=float(mcfg.get("pa_margin", 0.02)),
        compete=bool(mcfg.get("pa_compete", False)),
        norm_spatial=bool(mcfg.get("pa_norm_spatial", False)),
        num_classes=int(num_classes),
        grouped=bool(mcfg.get("pa_grouped", False)),
        res_scale=float(mcfg.get("pa_res_scale", 0.0)),
        mul=bool(mcfg.get("pa_mul", False)),
        mul_center=bool(mcfg.get("pa_mul_center", True)),
        mul_gate=float(mcfg.get("pa_mul_gate", -10.0)),
        mask_blur=bool(mcfg.get("pa_mask_blur", False)),
        readout=str(mcfg.get("pa_readout", "mean")),
        std_correction=int(mcfg.get("pa_std_correction", 0)),
        random_masks=bool(mcfg.get("pa_random_masks", False)),
        no_residual=bool(mcfg.get("pa_no_residual", False)),
        detach_feat=bool(mcfg.get("pa_detach", False)),
        perm_path=str(mcfg.get("pa_perm") or ""),
        kernel=int(mcfg.get("pa_kernel", 1)),
        js_w=float(mcfg.get("pa_js_w", 0.0)),
        js_target=float(mcfg.get("pa_js_target", 0.35)),
    )
    try:
        p = next(backbone.parameters())
        pa.to(device=p.device, dtype=p.dtype)
    except StopIteration:
        pass
    backbone.add_module("part_attn", pa)
    target.register_forward_hook(pa.hook)
    return pa


class PartFusion(nn.Module):
    """把 K 个部件特征（来自各自裁剪块的重新编码）做一层部件间自注意力后融合。

    与 PartAttention 的区别：这里的输入不是同一张特征图上池化出来的描述子，
    而是**每个部件单独裁出来、重新过骨干**得到的特征（MA-CNN / DCL 的做法）。
    单次前向的信息量因此不同：裁剪块被放大到 448，等于给部件更高的有效分辨率。

    零初始化输出，初始时融合分支不改变主分支。
    """

    def __init__(self, channels: int = 2048, dim: int = 256, depth: int = 1,
                 heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.proj = nn.Linear(channels, dim, bias=False)
        self.emb = nn.Parameter(torch.zeros(1, 32, dim))
        nn.init.trunc_normal_(self.emb, std=0.02)
        if depth > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        else:
            self.blocks = nn.Sequential(
                nn.Linear(dim, int(dim * mlp_ratio * 2)), nn.GELU(),
                nn.Linear(int(dim * mlp_ratio * 2), dim))
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z [B,K,C] -> [B,C]"""
        k = z.shape[1]
        t = self.proj(z) + self.emb[:, :k]
        t = self.blocks(t)
        return self.out(self.norm(t).mean(dim=1))


class SlotParts(nn.Module):
    """Slot Attention（Locatello et al., NeurIPS 2020）式的多部件提取。

    与 PartAttention 的差别在「怎么把部件分开」：
      * PartAttention：K 个 1×1 检测器各自出图，靠散度/竞争事后分开；
      * 这里：K 个可学习的 slot 作为查询，与所有空间 token 做**反向注意力**——
        每个 token 在 K 个 slot 之间做 softmax（跨 slot 竞争），再按 token 归一化，
        迭代 2–3 次用 GRU 更新 slot。相同 slot 无法占据同一个 token 的质量，
        这是结构性防塌缩，不是惩罚项。
    另外给 token 加 2D 正弦位置编码：没有它，空间上同质的 token 会有相同的 key，
    slot 必然收敛成同一个向量（官方实现在 flatten 前也加了位置编码）。

    接口与 PartAttention 对齐：hook 挂在一层的输出上、返回可加回特征图的残差、
    提供 part_masks（用于裁框）与 _last_part_logits（逐部件 CE）。
    """

    def __init__(self, channels: int = 2048, num_parts: int = 4, dim: int = 256,
                 iters: int = 3, heads: int = 4, mlp_ratio: float = 2.0,
                 dropout: float = 0.0, num_classes: int = 0, eps: float = 1e-8):
        super().__init__()
        self.num_parts = int(num_parts)
        self.dim = int(dim)
        self.iters = int(iters)
        self.eps = float(eps)
        # 与 PartAttention 对齐的接口属性（否则 forward/hook 第一行就 AttributeError）
        self.res_scale = 0.0
        self.no_residual = False
        self.detach_feat = False
        self.perm = None
        self.proj = nn.Linear(channels, dim, bias=False)
        self.slots_mu = nn.Parameter(torch.zeros(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.slots_mu, std=0.02)
        nn.init.trunc_normal_(self.slots_log_sigma, std=0.02)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim))
        self.norm_out = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.part_cls = nn.Linear(dim, int(num_classes)) if num_classes else None
        self._last_part_logits = None
        self._last_div = torch.zeros(())
        self._last_entropy = 0.0
        self._last_overlap = 0.0
        self._last_ratio = 0.0
        self._last_masks = None

    def _pos(self, h, w, device, dtype):
        return _sincos_2d(h, w, self.dim).to(device=device, dtype=dtype)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if self.detach_feat:
            feat = feat.detach()
        if self.perm is not None:
            feat = feat[:, self.perm]
        b, c, h, w = feat.shape
        n = h * w
        x = self.proj(feat.flatten(2).transpose(1, 2)) + self._pos(h, w, feat.device, feat.dtype)
        mu = self.slots_mu.expand(b, self.num_parts, self.dim)
        sigma = self.slots_log_sigma.exp().expand(b, self.num_parts, self.dim)
        slots = mu + sigma * torch.randn_like(mu)
        attn = None
        for _ in range(self.iters):
            q = self.q(slots) * (self.dim ** -0.5)            # [B,K,d]
            k = self.k(x)                                     # [B,N,d]
            v = self.v(x)
            logits = torch.einsum("bnd,bkd->bnk", k, q)       # [B,N,K]
            attn = logits.softmax(dim=-1)                     # 每个 token 在 K 个 slot 间竞争
            attn = attn + self.eps
            attn = attn / attn.sum(dim=1, keepdim=True)       # 再按 token 归一化
            # 负载均衡要用「归一化之前」的每个 slot 质量；归一化之后 sum 恒为 1，
            # 损失会退化成常数 (1-K/N)^2，梯度约为 0（审计发现）。
            updates = torch.einsum("bnk,bnd->bkd", attn, v)   # [B,K,d]
            slots = self.gru(updates.reshape(-1, self.dim),
                             slots.reshape(-1, self.dim)).reshape(b, self.num_parts, self.dim)
            slots = slots + self.mlp(self.norm_slots(slots))
        # 空间分布（用于裁框与诊断）：每个 slot 在 N 个位置上的质量
        a = attn.transpose(1, 2)                              # [B,K,N]（已按 token 归一化）
        a_norm = a / a.sum(-1, keepdim=True).clamp_min(self.eps)
        self._last_masks = a_norm.view(b, self.num_parts, h, w)
        # 负载均衡用**跨 slot softmax 之后、token 归一化之前**的质量：
        # raw_mass[b,k] = Σ_n softmax_slots(logits)[b,n,k]
        with torch.no_grad():
            raw = logits.softmax(dim=-1).sum(dim=1)            # [B,K]
            self._last_div = (self.num_parts * ((raw / n - 1.0 / self.num_parts) ** 2).sum(-1)).mean()
        with torch.no_grad():
            ent = -(a_norm * a_norm.clamp_min(1e-9).log()).sum(-1)
            self._last_entropy = float(ent.mean())
            af = a_norm / a_norm.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            sim = torch.einsum("bkn,bmn->bkm", af, af)
            kk = self.num_parts
            off = (sim.sum((1, 2)) - sim.diagonal(dim1=1, dim2=2).sum(1)) / max(kk * (kk - 1), 1)
            self._last_overlap = float(off.mean())
        y = self.out(self.norm_out(slots).mean(dim=1))
        if self.part_cls is not None:
            self._last_part_logits = self.part_cls(self.norm_out(slots))
        with torch.no_grad():
            g = feat.detach().flatten(2).norm(dim=2).mean()
            self._last_ratio = float(y.detach().norm().mean() / g.clamp(min=1e-12))
        return y

    def part_masks(self, feat: torch.Tensor) -> torch.Tensor:
        """跑一次前向拿到 [B,K,h,w] 的空间分布（用于裁框）。"""
        self.forward(feat)
        return self._last_masks

    def hook(self, module, inputs, output):
        y = self.forward(output)
        if self.no_residual:
            return output
        if self.res_scale > 0:
            # y 是 [B,C]（池化后的残差向量），不是 4D 特征图。
            # 用 s = r·g/(‖y‖+g)：初始 ‖y‖≈0 时 s≈r（有界），事后趋近 r·g/‖y‖。
            # 旧写法 r·g/‖y‖ 在零初始化时 s≈r·g/1e-6，反传放大约 4e5 倍（审计发现）。
            g = output.mean(dim=(2, 3)).norm(dim=1)                 # [B]
            yn = y.norm(dim=1)                                      # [B]
            s = (self.res_scale * g / (yn + g).clamp_min(1e-6)).detach()
            y = y * s[:, None]
        return output + y[:, :, None, None]


class MultiScaleParts(nn.Module):
    """多尺度 × 多区域组的部件 token 混合（GCL 的「跨尺度区域组 + 深层传播」简化版）。

    token 集合 = 2 个尺度 × K 个部件 = 2K 个 token：
      * layer3：28×28、1024 通道，分成 K 组（每组 256），检测 + 池化 → K 个 token；
      * layer4：14×14、2048 通道，分成 K 组（每组 512），检测 + 池化 → K 个 token。
    两套 token 各自投影到 d 维、加尺度嵌入，然后过 depth 层自注意力（可 2~3 层）。
    只有 layer4 的 token 通过残差回到特征空间（z' = z + back(attn)），因为下游的
    logit 级融合要用「通道组对齐」的描述子去切主分类器的权重；layer3 token 只提供上下文。

    初始化时 back 零初始化 => z' = z，与单尺度版逐位一致。
    """

    def __init__(self, ch3: int, ch4: int, num_parts: int = 4, dim: int = 256,
                 depth: int = 2, heads: int = 4, mlp_ratio: float = 2.0,
                 dropout: float = 0.0, num_classes: int = 0,
                 js_w: float = 0.0, js_target: float = 0.35):
        super().__init__()
        self.num_parts = int(num_parts)
        self.dim = int(dim)
        self.js_w = float(js_w)
        self.js_target = float(js_target)
        self.g3 = ch3 // self.num_parts
        self.g4 = ch4 // self.num_parts
        if ch3 % self.num_parts or ch4 % self.num_parts:
            raise ValueError("通道数必须能被 num_parts 整除")
        self.det3 = nn.Conv2d(ch3, self.num_parts, 1, groups=self.num_parts)
        self.det4 = nn.Conv2d(ch4, self.num_parts, 1, groups=self.num_parts)
        self.proj3 = nn.Linear(self.g3, dim, bias=False)
        self.proj4 = nn.Linear(self.g4, dim, bias=False)
        self.emb = nn.Parameter(torch.zeros(1, 2 * self.num_parts, dim))
        nn.init.trunc_normal_(self.emb, std=0.02)
        self.scale_emb = nn.Parameter(torch.zeros(1, 2, dim))
        nn.init.trunc_normal_(self.scale_emb, std=0.02)
        if depth > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        else:
            self.blocks = nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.back4 = nn.Linear(dim, self.g4)
        nn.init.zeros_(self.back4.weight)
        nn.init.zeros_(self.back4.bias)
        self._l3: torch.Tensor | None = None
        self._last_z: torch.Tensor | None = None
        self._last_part_logits = None
        self._last_div = torch.zeros(())
        self._last_entropy = 0.0
        self._last_overlap = 0.0
        self._last_ratio = 0.0

    @staticmethod
    def _group_pool(feat: torch.Tensor, det: nn.Conv2d, k: int, g: int) -> torch.Tensor:
        b, c, h, w = feat.shape
        m = torch.sigmoid(det(feat)).flatten(2)                 # [B,K,N]
        a = m / m.sum(-1, keepdim=True).clamp_min(1e-6)
        xs = feat.view(b, k, g, h, w)
        z = (xs * a.view(b, k, 1, h, w)).sum(dim=(3, 4))        # [B,K,g]
        return z

    def forward(self, l4: torch.Tensor) -> torch.Tensor:
        """返回 layer4 的混合后描述子 [B,K,g4]（不是残差）。"""
        l3 = self._l3
        z4 = self._group_pool(l4, self.det4, self.num_parts, self.g4)
        if l3 is None:
            self._last_z = z4
            return z4
        z3 = self._group_pool(l3, self.det3, self.num_parts, self.g3)
        t3 = self.proj3(z3) + self.emb[:, :self.num_parts] + self.scale_emb[:, :1]
        t4 = self.proj4(z4) + self.emb[:, self.num_parts:] + self.scale_emb[:, 1:2]
        t = torch.cat([t3, t4], dim=1)                          # [B,2K,d]
        t = self.blocks(t)
        t4p = self.norm(t[:, self.num_parts:])
        z4m = z4 + self.back4(t4p)                              # 初始 back=0 => z4m=z4
        self._last_z = z4m
        # 散度正则：对 layer4 的 K 张掩码算两两 JS（hinge 形式与 PartAttention 同标尺）。
        # ⚠️ 必须在 no_grad 之外算 —— 否则 _last_div 有数值但没有梯度，
        # pa_div_w / js_w 调多大都推不动检测器（审计 2026-10-29 发现）。
        if self.js_w > 0:
            m4 = torch.sigmoid(self.det4(l4)).flatten(2)
            a4 = m4 / m4.sum(-1, keepdim=True).clamp_min(1e-6)
            mm = (a4[:, :, None, :] + a4[:, None, :, :]) / 2.0
            la = a4.clamp_min(1e-9).log()[:, :, None, :]
            lm = mm.clamp_min(1e-9).log()
            kl1 = (a4[:, :, None, :] * (la - lm)).sum(-1)
            js = 0.5 * (kl1 + kl1.transpose(1, 2))
            off_ = 1.0 - torch.eye(js.size(1), device=js.device)
            js_off = (js * off_).sum(dim=(1, 2)) / (js.size(1) * (js.size(1) - 1))
            self._last_div = (self.js_target - js_off).clamp_min(0).mean() * self.js_w
        else:
            with torch.no_grad():
                self._last_div = torch.zeros((), device=l4.device)
        with torch.no_grad():
            self._last_ratio = float(self.back4(t4p).norm().mean() / z4.norm().mean().clamp_min(1e-12))
        return z4m

    def layer3_hook(self, module, inputs, output):
        self._l3 = output
        return None

    def hook(self, module, inputs, output):
        # 只计算并保存描述子，不改特征图（融合在 Classifier 的 logit 层做）
        self.forward(output)
        return output

    def part_masks(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.det4(feat))


def build_multiscale_parts(backbone: nn.Module, mcfg: dict, num_classes: int = 0):
    if getattr(backbone, "part_attn", None) is not None:
        raise RuntimeError("part_attn 已经注册过了")
    ch3 = backbone.layer3[-1].conv3.out_channels
    ch4 = backbone.layer4[-1].conv3.out_channels
    ms = MultiScaleParts(
        ch3=ch3, ch4=ch4,
        num_parts=int(mcfg.get("pa_parts", 4)),
        dim=int(mcfg.get("pa_dim", 256)),
        depth=int(mcfg.get("pa_depth", 2)),
        heads=int(mcfg.get("pa_heads", 4)),
        mlp_ratio=float(mcfg.get("pa_mlp", 2.0)),
        dropout=float(mcfg.get("pa_dropout", 0.0)),
        num_classes=int(num_classes),
        js_w=float(mcfg.get("pa_js_w", 0.0)),
        js_target=float(mcfg.get("pa_js_target", 0.35)),
    )
    try:
        p = next(backbone.parameters())
        ms.to(device=p.device, dtype=p.dtype)
    except StopIteration:
        pass
    backbone.add_module("part_attn", ms)
    backbone.layer3.register_forward_hook(ms.layer3_hook)
    backbone.layer4.register_forward_hook(ms.hook)
    return ms
