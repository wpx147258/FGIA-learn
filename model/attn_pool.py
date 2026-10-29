"""CNN 特征图 -> token 序列 -> 轻量自注意力（综述 V-D 指向的那类「部位间相互依赖」）。

动机：
  综述里那段原话是 "jointly learn the interdependencies among part-level features"。
  我们此前所有部位方法（三流、CAP、NetVLAD、CAM 裁剪）都是「把某处取出来再用」，
  没有一个是让部位之间互相看。这个模块做的就是让 49 个位置互相看。

设计上有一条硬约束：**初始时必须等于基线**。
  输出层零初始化，所以训练第 0 步时模块贡献为 0，模型和 89.40 那份完全等价。
  这样训练后的任何变化都只能来自这个模块本身，而不是"多了一层初始化扰动"。

数据流（以 448 输入、layer4 输出 [B,2048,14,14] 为例）：
    feat [B,2048,14,14]
      -> 2x2 平均池化      [B,2048,7,7]      （tokens=49 时）
      -> 1x1 卷积降维      [B,dim,7,7]
      -> flatten + 转置    [B,49,dim]
      -> 加可学习位置编码
      -> N 个 Transformer block
      -> token 平均         [B,dim]
      -> 线性层回到 2048    [B,2048]   （零初始化）
    最终特征 = GAP(feat) + 上式         （残差相加，不是拼接）

为什么用相加而不是拼接：拼接会改变分类头输入维度，等于换了一个头；
相加保持 2048 维，头完全不动，基线的参数量与结构不变。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttnPool(nn.Module):
    """把特征图压成 token 序列做自注意力，输出与池化特征同维的残差。"""
    ### Q ：采用 readout: = "cls" 是否建议depth>1?

    def __init__(self, in_channels: int = 2048, out_dim: int = 2048, dim: int = 768,
                 tokens: int = 196, depth: int = 1, heads: int = 8,
                 mlp_ratio: float = 2.0, dropout: float = 0.0, use_pos: bool = False,
                 readout: str = "mean", replace: bool = False):
        super().__init__()
        self.tokens = int(tokens)
        self.use_pos = bool(use_pos)
        if readout not in ("mean", "cls", "gate"):
            raise ValueError(f"readout 只能是 mean | cls | gate，收到 {readout!r}")
        self.readout = readout
        # gate 读出：pooled = GAP + Σ_n g_n·out(t_n)，g_n = sigmoid(MLP(t_n))，
        # 末层初始 −10 → g≈4.5e-5，且 out 零初始化，初始输出精确等于 GAP。
        # 与 TokenLearner 的门控同一设计：模型可以「不用 token」直接退回全局平均。
        self.gate_mlp = None
        if readout == "gate":
            self.gate_mlp = nn.Sequential(
                nn.Linear(dim, max(64, dim // 8)), nn.GELU(),
                nn.Linear(max(64, dim // 8), 1))
            nn.init.zeros_(self.gate_mlp[-1].weight)
            nn.init.constant_(self.gate_mlp[-1].bias, -10.0)
        # 49 = 先在特征图上做 2x2 平均池化；其余（196 / 784）= 不池化，直接用全部位置。
        # 文献调研的结论：细粒度上主流是 1x1 降维后 flatten 全部位置，
        # 没有工作用 2x2 池化到 49，所以 49 只作为最后面对照保留。
        self.pre_pool = nn.AvgPool2d(2) if tokens == 49 else nn.Identity()
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=1)
        if self.use_pos:
            self.pos = nn.Parameter(torch.zeros(1, tokens, dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
        else:
            self.pos = None

        if depth > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        else:
            # depth=0 是控制臂：不接注意力，只保留「1x1 投影 -> token 平均 -> 线性」。
            # 用来分离「多出来的参数与池化」和「注意力本身」的贡献。
            self.blocks = nn.Identity()
        self.norm = nn.LayerNorm(dim)

        # 两种出口：
        #   residual（默认）：零初始化，输出作为**残差**加回 GAP —— 初始等价基线，
        #                     但多出一条通往分类器的通路；
        #   replace：直接把注意力池化的结果**替换** GAP，用 back=pinv(proj) 把 dim 维
        #            投影回 C 维，并用 alpha 在 back(v) 与 GAP 之间混合（初始 0.5，
        #            此时 back(v)≈GAP，所以初始仍等于 GAP）。用来检验「额外残差通路
        #            本身」是否有害。
        self.replace = bool(replace)
        if self.replace:
            # ⚠️ 不可行：proj 是 C->dim 的降维（dim=256 < C=2048），从 dim 维还原 C 维
            # 只是最小二乘投影，初始输出与 GAP 的差距约 0.27（logit 尺度），
            # 不是「初始等于 GAP」。要精确替换 GAP，必须把注意力权重直接作用在
            # C 维原始特征上（见 TokenLearner 的 replace 分支）。
            raise ValueError("AttnPool 的 replace 模式不可用（dim < C 无法还原 GAP）；"
                             "请用 MODEL.tokenlearner + tl_replace 或 MODEL.content_pool")
        else:
            self.out = nn.Linear(dim, out_dim)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat [B,C,H,W] -> [B,C]，与 GAP 输出同维，供直接相加。"""
        x = self.pre_pool(feat)
        x = self.proj(x).flatten(2).transpose(1, 2)     # [B,N,dim]
        if self.readout == "cls":
            # 全局 token：初始 = N 个 token 的平均（GAP），拼在序列最前面。
            # 它不带位置编码（是全局表示，不是某个空间格子）；经过自注意力后
            # 吸收其它 token 的信息，读出它自己作为聚合结果。
            gap = x.mean(dim=1)                          # [B,dim]
            if self.pos is not None:
                x = x + self.pos[:, : x.size(1)]
            t = torch.cat([gap.unsqueeze(1), x], dim=1)  # [B,N+1,dim]
            t = self.blocks(t)
            x = self.norm(t)[:, 0]                       # 读第 0 个 token（更新后）
            t_mean = t[:, 0]
        else:
            if self.pos is not None:
                x = x + self.pos[:, : x.size(1)]
            t = self.blocks(x)
            x = self.norm(t).mean(dim=1)                 # token 平均
            t_mean = t.mean(dim=1)
        if self.replace:
            # 用未过 LayerNorm 的平均 token：此时它正是 proj(GAP)，back 的伪逆能还原 GAP
            t_all = t if self.readout == "cls" else None
            x_raw = (t_all[:, 0] if t_all is not None else t_mean)
            gap = feat.mean(dim=(2, 3))
            a = torch.sigmoid(self.alpha)
            y = a * self.back(x_raw) + (1.0 - a) * gap
        elif self.gate_mlp is not None:
            tn = self.norm(t)                                  # 与 mean 读出一致，先过 LayerNorm
            v = self.out(tn)                                  # [B,N,C]
            g = torch.sigmoid(self.gate_mlp(tn))              # [B,N,1]
            y = (g * v).sum(dim=1)                            # [B,C]，初始 out 全零 → y=0
        else:
            y = self.out(x)
        # 记两个量给日志：模块输出的绝对范数，以及它与 GAP 特征的相对幅度。
        # 后者才是关键 —— 残差要从 0 长到和 GAP 可比的数量级，才可能真正影响分类。
        # 训练和推理都记：否则指标要等第一次验证才出现，CSV 表头会缺列
        with torch.no_grad():
            g = feat.mean(dim=(2, 3))
            self._last_out_norm = float(y.detach().norm(dim=1).mean())
            self._last_ratio = float(
                (y.detach().norm(dim=1).mean() / g.norm(dim=1).mean().clamp(min=1e-12)))
        return y


class SEGate(nn.Module):
    """残差式通道门控：out = feat * (1 + delta)，delta 初始 0。

    旧版是绝对门控（out = feat * sigmoid(...)），乘在完整特征上、没有捷径保底，
    门学偏会整体缩放特征。这里改成**乘性残差**：缩放因子 = 1 + delta，delta 从 0
    学起，初始恒等（等价基线），学到的是「相对恒等的缩放增量」。
    delta 用 tanh 限制在 (-1,1)，保证 1+delta ∈ (0,2) 恒为正，通道不会被翻转为负。

    为什么加在最终特征图上而不是插进每个 bottleneck：标准 SE-ResNet（残差分支上）
    现在由 MODEL.se / backbone.add_se_blocks 提供；这里只做最终特征图上的一次
    乘性残差门控，作为更轻的对照。
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(8, channels // reduction)
        self.reduce = nn.Conv2d(channels, mid, 1)
        self.expand = nn.Conv2d(mid, channels, 1)
        # delta 初始 0：expand 零初始化 -> tanh(0)=0 -> 1+delta=1（恒等）
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        s = feat.mean(dim=(2, 3), keepdim=True)
        delta = torch.tanh(self.expand(torch.relu(self.reduce(s))))
        return feat * (1.0 + delta)


class TokenLearner(nn.Module):
    """软空间注意力池化（TokenLearner, NeurIPS 2021）。

    与 AttnPool 的区别：AttnPool 让 196 个**空间格子**互相看，输出仍是"整体的一个向量"；
    这里则是学 S 张空间权重图，每张图把 196 个位置加权求和成一个 token，得到 S 个
    "部位级"表示。默认（attn_depth=0）这些 token 之间不交互、直接平均；设
    attn_depth>0 可选让它们互相看（与 AttnPool 的对照项）。

    为什么值得试：我们的诊断显示
      - 61.9% 的 token 对真实类贡献为负，但硬剪掉它们不改变准确率（负贡献量级太小）
      - 贡献最高的 20% token 占正向质量 96.9%
    硬剪无效，但"按可学习的空间权重重新加权"未必无效 —— 这是两者唯一没被排除的差别。

    输出层零初始化，训练第 0 步与基线等价。

    ⚠️ 两种不同的「塌缩」，必须用两个不同的量区分（2026-10-29 修正）：
      1) 每张图塌成一个点：熵 → 0。各图可选中相同或不同位置；每个 token 几乎只取
         一个位置的特征，但不代表特征本身没有信息（实测 r50_tl_s8：熵 0.0076、两两 JS 0.57）。
      2) 八张图变成同一张图的副本：熵可以「看起来很健康」（=2.0），但两两 JS → 0
         （实测 r50_tl_s8_reg：熵 2.03、JS 0.047、中心化余弦 0.86）。
    只看熵无法识别第 2 种。两个量的解释：
      * 均匀分配到 k 个位置时 H=ln k；一般分布的 exp(H) 是有效位置数，
        不等于实际支撑大小，也不保证空间连续。目标 H=2.1~2.8 可作为实验起点。
      * JS 是两分布的 Jensen-Shannon 散度（自然对数，上界 ln2≈0.693）：
        分布相同为 0，支撑完全不重叠为 ln2。仅当两图各自在同样多的位置上均匀分布时，
        共享一半支撑才对应 JS≈0.35；一般不能将 JS 阈值直接换算为重叠比例。
        当前正则约束每张图片的图对平均 JS，不保证每一对都达到 js_target。

    历史 bug（已修，两处）：
      * 原 einsum("bns,bms->bsm") 对 n 求和、输出 [B,S,N]；因每张图的权重和为 1，
        结果等于 w.transpose(1,2)，没有计算图间重叠；已改为 `w.transpose(1,2) @ w`；
      * `div_w` 只被当作开关，数值上没有乘进正则；已改为 `div_w * 重叠均值`。
    旧正则仍可能影响梯度，但不符合图间重叠惩罚的设计。r50_tl_s8 测试了无正则版本，
    r50_tl_s8_reg 测试了旧正则版本；两者均不能替代修正后正则的实验。
    """

    def __init__(self, in_channels: int = 2048, out_dim: int = 2048, dim: int = 256,
                 s_tokens: int = 8, dropout: float = 0.0,
                 ent_target: float = 0.0, div_w: float = 0.0,
                 js_w: float = 0.0, js_target: float = 0.35, js_reduce: str = "mean",
                 ent_mode: str = "target", ent_floor: float = 0.5,
                 ent_ceiling: float = 0.0, ent_ramp: float = 0.0,
                 attn_depth: int = 0, attn_heads: int = 4,
                 readout: str = "mean", replace: bool = False):
        super().__init__()
        self.s = int(s_tokens)
        self.dim = int(dim)
        self.proj = nn.Conv2d(in_channels, dim, 1)
        self.score = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, self.s),
        )
        self.norm = nn.LayerNorm(dim)
        self.replace = bool(replace)
        if readout not in ("mean", "cls", "concat", "gate"):
            raise ValueError(f"readout 只能是 mean | cls | concat | gate，收到 {readout!r}")
        self.cat = None
        self.gate_mlp = None
        if self.replace:
            # concat 读出：S 个 token 各自投影到 C/S 维，再拼起来（总维仍是 C，
            # 分类头大小与基线完全相同）。分类头于是能**对每个部件分别加权**，
            # 而不是先把它们平均掉 —— 这是 mean 读出不具备的信息通道。
            if int(attn_depth) > 0 or readout not in ("mean", "concat", "gate"):
                raise ValueError("TokenLearner 的 replace 只支持 attn_depth=0 且 readout=mean|concat|gate")
            if readout == "gate":
                # 逐 token 门控读出：pooled = GAP + Σ_s g_s·z_s，
                # g_s 由每个 token 自己的特征算出来（逐样本动态），末层初始化 −10 → g≈4.5e-5，
                # 初始输出≈GAP。好处是部件不可靠时模型可以**把门关掉退回 GAP**，
                # 而不是被迫平均——这正是 mean 读出缺的那条退路。
                self.gate_mlp = nn.Sequential(
                    nn.Linear(in_channels, max(64, in_channels // 8)), nn.GELU(),
                    nn.Linear(max(64, in_channels // 8), 1))
                nn.init.zeros_(self.gate_mlp[-1].weight)
                nn.init.constant_(self.gate_mlp[-1].bias, -10.0)
            if readout == "concat":
                if in_channels % self.s != 0:
                    raise ValueError(
                        f"readout=concat 要求通道数 {in_channels} 能被 S={self.s} 整除")
                self.cat = nn.Linear(in_channels, in_channels // self.s)
            # 替换式池化：学到的空间权重直接作用在 C 维原始特征上，输出就在 C 空间。
            # score 末层置零 => 初始权重均匀 => 输出逐位等于 GAP（唯一能精确做到的结构）。
            self.out = None
            nn.init.zeros_(self.score[-1].weight)
            nn.init.zeros_(self.score[-1].bias)
        else:
            self.out = nn.Linear(dim, out_dim)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        self._last_entropy = None
        # replace 出口下缓存 [B,S,C] 的部件 token（C = 骨干通道），供分类头做逐 token
        # 辅助 CE（MA-CNN 式的「每个部件都要能分类」）。非 replace 出口的 token 在
        # dim 空间里，分类头吃不了，保持 None。
        self._last_z = None
        # 防塌缩有两条独立的轴，别混用：
        #   ent_target —— 管单张图的集中程度；均匀覆盖 k 个位置时 H=ln k。
        #                 一般用 exp(H) 表示有效位置数；H=2.0~2.5 对应约 7.4~12.2，
        #                 H=0.5 对应约 1.6。这不代表实际覆盖数量或空间连续性。
        #   div_w / js_w —— 约束图间重叠或分布差异；仅匹配目标熵不能保证它们分化
        #                 （实测把熵拉到 2.0 后，八张图仍可能是同一张的副本）。
        # 旧 div 的 einsum 未计算图间重叠，div_w 也未乘入损失；并非完全无梯度，见类 docstring。
        self.ent_target = float(ent_target)
        # ent_mode:
        #   "target" —— 双侧惩罚 (ent − target)²。⚠️ 它同时会「把太尖的图推开」，
        #               在残差出口下会把部件推到背景上去（实测富集比 0.30×）。
        #   "floor"  —— 单侧惩罚 relu(floor − ent)²。只防「塌成点」，不强迫摊开，
        #               更适合「让 CE 决定部件在哪、正则只兜底」。
        if ent_mode not in ("target", "floor", "ceiling"):
            raise ValueError(f"ent_mode 只能是 target | floor | ceiling，收到 {ent_mode!r}")
        self.ent_mode = ent_mode
        self.ent_floor = float(ent_floor)
        # ceiling —— 只惩罚「摊得太开」的单侧上限：relu(ent − ceiling)²。
        # 与 floor（只防塌缩）互为镜像。用户判断：权重应当集中（部件是局部区域），
        # 而不该摊到 ~7 格以上；同时允许图与图之间大幅重叠（JS 目标可以很低）。
        self.ent_ceiling = float(ent_ceiling)
        # ent_ramp —— 上限的退火比例：训练前 ent_ramp（占总轮数的比例）内，
        # 有效上限从 ln N（等于不约束）线性降到 ent_ceiling。
        # 动机：替换出口下上限直接生效时，初始化熵 ln196≈5.28 与 0.5 的差是 22.8，
        # 会像双侧 target 那样从第 0 步改写分类输入；退火让 CE 先把掩码塑形。
        self.ent_ramp = float(ent_ramp)
        self.progress = 1.0     # 由 trainer 每轮写入 0..1；1 = 退火结束
        self.div_w = float(div_w)
        # JS 正则：当每张图片的图对平均 JS 低于 js_target 时施加惩罚。
        # 分布相同 JS=0，支撑完全不重叠 JS=ln2；仅对支撑大小相同的均匀分布，
        # 共享 25% / 50% 支撑才分别对应约 0.52 / 0.35，不能推广为一般的重叠比例。
        self.js_w = float(js_w)
        self.js_target = float(js_target)
        # js_reduce 决定「图对平均」在 hinge 的内侧还是外侧：
        #   "mean"（旧行为）—— hinge(mean_ij JS_ij − target)。均值可以被少数差异很大的
        #       图对抬上去，剩下的图对即使完全相同也不受惩罚。实测 tl_floor 的
        #       train_tl_reg 到 ep15 就掉到 1e-4，正是这条捷径。
        #   "pair"—— mean_i hinge(mean_{j≠i} JS_ij − target)。每一张图都要和其余图
        #       拉开，不能靠两三张图把平均顶上去。
        if js_reduce not in ("mean", "pair"):
            raise ValueError(f"js_reduce 只能是 mean | pair，收到 {js_reduce!r}")
        self.js_reduce = js_reduce
        # 部件 token 之间做自注意力（0 = 不做，保持旧行为）。
        # readout="cls" 与 AttnPool 对齐：用「token 均值」当全局 token 拼在最前面，
        # 过完注意力后读它自己 —— 两种池化机制可以形成干净对照。
        self.readout = readout
        self.attn_depth = int(attn_depth)
        if attn_depth > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=attn_heads, dim_feedforward=int(dim * 2),
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, num_layers=int(attn_depth))
        else:
            self.blocks = nn.Identity()
        self._reg = None

    def set_progress(self, p: float):
        """训练进度 0..1，用于上限退火（ent_ramp）。"""
        self.progress = float(min(max(p, 0.0), 1.0))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.proj(feat).flatten(2).transpose(1, 2)      # [B,N,dim]
        logits = self.score(x)                              # [B,N,S]
        w = logits.softmax(dim=1)                           # 对位置归一化
        z = torch.einsum("bns,bnd->bsd", w, x)              # [B,S,dim]  每张图一个 token
        if self.readout == "cls":
            gap = z.mean(dim=1, keepdim=True)               # 每次前向用部件 token 均值构造全局 token
            t = self.blocks(torch.cat([gap, z], dim=1))
            v = self.norm(t)[:, 0]
        else:
            t = self.blocks(z)
            v = self.norm(t).mean(dim=1)
        # 每张图在 N 个位置上的熵 [B,S]：衡量权重的集中程度，不包含空间几何信息，
        # 也不说明 S 张图彼此是否相同；后者可用 JS 衡量，div 则是内积重叠惩罚。
        ent = -(w * w.clamp_min(1e-9).log()).sum(dim=1)
        with torch.no_grad():
            self._last_entropy = float(ent.mean())
        reg = None
        if self.ent_mode == "floor" and self.ent_floor > 0:
            reg = (self.ent_floor - ent).clamp_min(0).pow(2).mean()
        elif self.ent_mode == "ceiling" and self.ent_ceiling > 0:
            # 只压「太散」，不推「更尖」：已经比 ceiling 更集中时梯度为 0，
            # 不会像双侧 target 那样把已经很尖的图反向推回去。
            ceil = self.ent_ceiling
            if self.ent_ramp > 0 and self.progress < self.ent_ramp:
                import math as _m
                ln_n = _m.log(float(w.shape[1]))
                frac = self.progress / self.ent_ramp          # 0 → 1
                ceil = ln_n + (ceil - ln_n) * frac
            reg = (ent - ceil).clamp_min(0).pow(2).mean()
        elif self.ent_target > 0:
            reg = ((ent - self.ent_target) ** 2).mean()
        if self.div_w > 0:
            # ov[b,i,j] = sum_n w[b,n,i] * w[b,n,j]，形状 [B,S,S]。
            # 概率分布的内积在 [0,1] 内：支撑不重叠为 0，同一位置上的单点分布为 1；
            # 两张相同的均匀图内积仅为 1/N，所以不能将内积直接解释为分布相似度。
            # 旧 einsum("bns,bms->bsm") 输出 [B,S,N]，等于转置后的 w，未计算图间重叠。
            ov = w.transpose(1, 2) @ w
            s_ = ov.size(1)
            off = (ov.sum(dim=(1, 2)) - ov.diagonal(dim1=1, dim2=2).sum(1)) / (s_ * (s_ - 1))
            # ⚠️ 这里原来是 `dv = off.mean(); reg += dv` —— div_w 只当开关用，
            # 数值上完全没有乘进去。现在按权重乘，div_w 才真正控制强度。
            dv = self.div_w * off.mean()
            reg = dv if reg is None else reg + dv
        if self.js_w > 0 and self.s > 1:
            # 沿位置维 N 计算两两 JS：JS(P,Q) = (KL(P||M) + KL(Q||M))/2，M=(P+Q)/2。
            # 精确计算时 JS=0 当且仅当分布相同，上界为 ln2；clamp 用于避免 log(0)。
            m_ = (w[:, :, :, None] + w[:, :, None, :]) / 2.0
            lw = w.clamp_min(1e-9).log()[:, :, :, None]
            lm = m_.clamp_min(1e-9).log()
            kl1 = (w[:, :, :, None] * (lw - lm)).sum(1)          # [B,S,S]
            js = 0.5 * (kl1 + kl1.transpose(1, 2))
            off_ = 1.0 - torch.eye(js.size(1), device=js.device)
            if self.js_reduce == "pair":
                # 每张图与其余图的平均 JS 各自过 hinge 再平均：每张图都要被拉开，
                # 不能靠少数图对把整体均值顶上去（旧写法在这点上可以被绕过）。
                per_map = (js * off_).sum(dim=2) / max(js.size(1) - 1, 1)   # [B,S]
                jr = self.js_w * (self.js_target - per_map).clamp_min(0).mean()
            else:
                # 旧行为：先排除自身比较并平均图对，再做 hinge。
                js_off = (js * off_).sum(dim=(1, 2)) / (js.size(1) * (js.size(1) - 1))
                jr = self.js_w * (self.js_target - js_off).clamp_min(0).mean()
            reg = jr if reg is None else reg + jr
        self._reg = reg
        if self.replace:
            # w [B,N,S] 直接加权 C 维原始特征；初始 w 均匀 => 输出 == GAP
            f_c = feat.flatten(2).transpose(1, 2)                # [B,N,C]
            z_c = torch.einsum("bns,bnc->bsc", w, f_c)           # [B,S,C]
            self._last_z = z_c
            if self.gate_mlp is not None:
                gap = feat.mean(dim=(2, 3))                       # [B,C]
                g = torch.sigmoid(self.gate_mlp(z_c))             # [B,S,1]
                return gap + (g * z_c).sum(dim=1)
            if self.cat is not None:
                # [B,S,C] -> [B,S,C/S] -> [B,C]：部件各自占一段通道，分类头可分别加权
                return self.cat(z_c).reshape(z_c.shape[0], -1)
            return z_c.mean(dim=1)
        return self.out(v)


class ContentPool(nn.Module):
    """内容相关的空间注意力池化，替换 GAP，输入 [B,C,H,W]、输出 [B,C]。

    与 AttnPool / TokenLearner 的残差分支不同，本模块直接返回完整池化特征。
    value 使用原始 C 维特征，仅用低维 key 和共享 query 生成一张空间权重图，
    不做位置间自注意力，也不对聚合后的特征做输出投影。

    初始化保留 Wk 的随机权重，只将 q 置零：logits=0，注意力均匀，数学上输出
    等于 GAP（浮点运算可能有舍入误差）。第一步 q 可获得梯度，Wk 的分类损失
    梯度为零；q 更新后 Wk 才能获得梯度。不能同时把 q 和 Wk 置零，否则两者
    梯度都为零，注意力会一直停在均匀分布。alpha 在 pool 与 GAP 不同时才能学习。

    结构（参数量 C×r + r + 1）：
        k    = X Wk                     [B,N,r]
        attn = softmax(q·k / sqrt(r))   [B,N]
        pool = attn @ X                 [B,C]   （均匀时 == GAP）
        out  = a * pool + (1-a) * GAP   a = sigmoid(alpha)，初始 0.5
    """

    def __init__(self, channels: int, rank: int = 64):
        super().__init__()
        self.channels = int(channels)
        self.rank = int(rank)
        if self.channels <= 0 or self.rank <= 0:
            raise ValueError("ContentPool 的 channels 和 rank 必须为正整数")
        self.wk = nn.Linear(self.channels, self.rank, bias=False)
        self.q = nn.Parameter(torch.zeros(self.rank))
        self.alpha = nn.Parameter(torch.zeros(1))     # sigmoid(0)=0.5
        # Wk 保持默认随机初始化，避免与零 query 相乘导致两侧梯度同时为零。
        self._last_entropy = 0.0

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 4 or feat.size(1) != self.channels:
            raise ValueError(
                f"ContentPool 需要 [B,{self.channels},H,W]，收到 {tuple(feat.shape)}")
        if feat.size(2) == 0 or feat.size(3) == 0:
            raise ValueError("ContentPool 的空间维度 H 和 W 必须非空")
        x = feat.flatten(2).transpose(1, 2)                  # [B,N,C]
        k = self.wk(x)                                      # [B,N,r]
        # AMP 下以 FP32 计算打分、softmax、累加与熵，避免低精度下的溢出和 log(0)。
        # 保留 FP64 输入的精度；最终输出恢复输入 dtype，供主干的分类头直接使用。
        acc_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        x_acc = x.to(dtype=acc_dtype)
        logits = (k.to(dtype=acc_dtype) * self.q.to(dtype=acc_dtype)).sum(-1)
        logits = logits / (self.rank ** 0.5)                 # [B,N]
        attn = logits.softmax(dim=-1)
        pool = (attn.unsqueeze(-1) * x_acc).sum(dim=1)       # [B,C]
        gap = x_acc.mean(dim=1)
        a = torch.sigmoid(self.alpha.to(dtype=acc_dtype))
        with torch.no_grad():
            ent = -(attn * attn.clamp_min(1e-9).log()).sum(1).mean()
            self._last_entropy = float(ent)
        return (a * pool + (1.0 - a) * gap).to(dtype=feat.dtype)
