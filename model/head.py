"""分类头。

现在只有线性头。后续要加属性头 / 部位头，都写在这个文件里，
然后在 HEADS 里登记一行，config 里的 `MODEL.head` 就能选到。
"""

from __future__ import annotations

from model.cap_head import CAPHead

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    """线性分类头：backbone 特征 -> 类别 logits。

    norm=True 时在 FC 前加一层 BatchNorm1d。动机：不同预训练权重的特征幅度
    差别很大（实测 a1 的 GAP 特征 rms 0.10，SWSL 是 0.65，6.5 倍），而 FC 的
    梯度正比于特征幅度。全局梯度裁剪之后，backbone 的梯度占比被放大，于是
    「用 lr=1e-3 训 SWSL」会把预训练特征直接推平、loss 卡在 ln(200)。
    BN 把输入拉回单位方差，backbone 的梯度就不再随特征尺度漂移。
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0,
                 norm: bool = False):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_features) if norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.norm(x)))

    def cam_weight(self, feat_dim: int) -> torch.Tensor:
        """CAM 用的分类权重 [num_classes, C]，和 backbone 特征图的通道维对齐。"""
        return self.fc.weight


# 名字 -> 构造器。加新头在这里加一行即可。





class GAPStdHead(nn.Module):
    """GAP + 空间标准差拼接 —— 二阶统计量最便宜的近似。

    动机：GAP 每个通道只留一个均值，丢掉了「这个通道在图上变化有多大」。
    标准差是方差的平方根，也就是二阶矩的**对角部分**，正好是第 4.2 节
    bilinear pooling 想编码的东西的最粗版本，但没有外积那个 2048×2048 的维度爆炸
    （Covariance pooling 是完整版，参数/维度都贵得多）。

    特征图 [B,C,h,w] → [B,2C] → FC。参数 819K，是线性头的两倍。
    评估路径和训练路径完全一样，没有额外的分支。
    """

    on_feature_map = True

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(2 * in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=(2, 3))                      # 一阶矩
        # unbiased=False：h×w=196 时差别可忽略，但特征图退化成 1×1 时不会出 NaN
        sd = x.std(dim=(2, 3), unbiased=False)       # 二阶矩的对角
        return self.fc(self.dropout(torch.cat([mu, sd], dim=1)))

    def cam_weight(self, feat_dim: int) -> torch.Tensor:
        """CAM 只用得上均值那半权重：标准差没有空间位置，无法投影回特征图。"""
        w = self.fc.weight
        if w.shape[1] == 2 * feat_dim:
            return w[:, :feat_dim]
        return w


class CompactBilinearHead(nn.Module):
    """Compact Bilinear Pooling（Gao et al., CVPR 2016）—— Tensor Sketch 版。

    Bilinear CNN 的做法：每个空间位置取通道向量 v（C 维），算外积 v vᵀ 再对位置求和，
    得到 C×C 的二阶统计量。它看到的是**通道对之间的协方差**，而不是每个通道自己的方差
    （后者就是 GAPStdHead 的 std 分支，只有对角线）。

    直接展开 C×C = 2048² 没法用，所以论文用 Tensor Sketch（Pham & Pagh 2013）把它
    近似成 d 维：

        φ(v) = FFT( count_sketch(v; h, s) )
        bilinear(v) ≈ IFFT( φ₁(v) ⊙ φ₂(v) )

    两路用**独立**的 (h, s)，否则 h₁=h₂ 时结果退化。整个过程没有可学习参数，
    h/s 是固定的随机向量（注册成 buffer，不进 state_dict）。

    之后按 B-CNN 的标准做法：空间求和 → signed sqrt → L2 归一化 → FC。
    """

    on_feature_map = True

    def __init__(self, in_features: int, out_features: int,
                 dim: int = 8192, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        g = torch.Generator().manual_seed(0)
        for name in ("h1", "s1", "h2", "s2"):
            if name.startswith("h"):
                buf = torch.randint(0, self.dim, (in_features,), generator=g)
            else:
                buf = torch.randint(0, 2, (in_features,), generator=g) * 2 - 1
            self.register_buffer(name, buf, persistent=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.dim, out_features)

    def _sketch(self, v: torch.Tensor, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """count sketch：[B,C,N] → [B,dim,N]，把每个通道按 h 打散到 dim 个格子并乘符号 s。"""
        B, C, N = v.shape
        out = v.new_zeros(B, self.dim, N)
        idx = h.view(1, C, 1).expand(B, C, N)
        out.scatter_add_(1, idx, v * s.view(1, C, 1))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # FFT 强制走 fp32：bf16 下 torch.fft 会退到 ComplexHalf（实验性实现，
        # 会打警告，精度和算子覆盖都不完整）。这里只多占一点显存。
        v = x.reshape(B, C, H * W).float()
        fx = torch.fft.rfft(self._sketch(v, self.h1, self.s1), dim=1)
        fy = torch.fft.rfft(self._sketch(v, self.h2, self.s2), dim=1)
        pos = torch.fft.irfft(fx * fy, n=self.dim, dim=1)      # [B,dim,N]
        pooled = pos.sum(dim=2)                                # 空间求和
        # signed sqrt + L2：B-CNN 的标准后处理，压低大幅值维度、抬升小幅值维度。
        # ⚠️ 必须加 eps：sqrt 在 0 处的导数是 inf，只要有一个元素恰好等于 0，
        # 反向就会得到 inf*0 = NaN 并污染整条链路（实测跑到第二轮的某个 batch 就崩）。
        pooled = torch.sign(pooled) * (pooled.abs() + 1e-8).sqrt()
        pooled = torch.nn.functional.normalize(pooled, dim=1)
        # 给 CAM 辅助分支留一份**逐位置**的贡献（训练时才留，约 100MB @ batch16）。
        # 双线性没有「每通道一个权重」，CAM 只能这样算：位置 p 对第 c 类的贡献
        # = W_c · φ(v_p)，所以要把逐位置的 φ 留下来。
        self._pos = (torch.sign(pos) * pos.abs().sqrt()).detach() if self.training else None
        return self.fc(self.dropout(pooled))

    def cam_weight(self, feat_dim: int) -> torch.Tensor:
        """双线性没有逐通道权重；CAM 走逐位置贡献那条路（见 _pos）。"""
        raise NotImplementedError(
            "CompactBilinearHead 的 CAM 请用逐位置贡献（_pos），不用 cam_weight")


class NetVLADHead(nn.Module):
    """NetVLAD 池化 + 分类，替代 GAP + Linear。

    动机（来自 CAP, AAAI 2021 的消融）：GAP 把整张特征图平均成一个向量，
    丢掉了「特征在空间上怎么分布」。NetVLAD 把特征聚成 K 个簇，记录每个簇
    相对质心的残差和 —— 相当于保留了这张图的特征分布形状。
    论文里 CUB 上这一步（配合 LSTM）单独贡献 +4.4（Xception, 弱 baseline）。

    和原始 NetVLAD 的差异：
      * 先用 1x1 conv 把 2048 维降到 downsample 维。原始版本 K×D×num_classes
        的 FC 是 32×2048×200 = 13M 参数，对 5395 张训练图太大。
        我们之前测过：2048→256 的线性探针只掉 0.43 个点，降维代价很小。
      * 没接 LSTM（CAP 用它编码区域顺序，我们这里是单张特征图，没有区域序列）。

    ⚠️ 它吃的是**池化前**的特征图 [B, C, H, W]，不是 [B, C]。
       Classifier 看到 on_feature_map=True 就会改走 backbone.forward_features。
    """

    on_feature_map = True

    def __init__(self, in_features: int, out_features: int,
                 num_clusters: int = 32, downsample: int = 256,
                 dropout: float = 0.0, assign_init_scale: float = 60.0):
        super().__init__()
        d = int(downsample or in_features)
        self.proj = nn.Conv2d(in_features, d, 1) if d != in_features else nn.Identity()
        # soft assignment：每个空间位置属于各簇的概率
        self.assign = nn.Conv2d(d, num_clusters, 1)
        # ⚠️ 关键：默认初始化的 1x1 conv 输出接近 0，softmax 直接饱和成 1/K ——
        # 也就是「所有簇平分每个位置」，此时 V_k ≈ GAP 的倍数，32 个簇完全冗余。
        # 实测训练完 logits std 只有 0.18，熵 99.5% × ln(K)。
        # 把权重放大，让优化从「有结构」的点出发，而不是从均匀这个吸引子出发。
        # 60 倍让初始化后的熵从 ~100% 降到 ~62%（实测 20 倍只到 85%，不够）。
        with torch.no_grad():
            self.assign.weight.mul_(assign_init_scale)
            if self.assign.bias is not None:
                self.assign.bias.mul_(assign_init_scale)
        self.centers = nn.Parameter(torch.randn(num_clusters, d) * 0.1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(num_clusters * d, out_features)
        self.num_clusters, self.dim = num_clusters, d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                                  # [B, d, H, W]
        B, d, H, W = x.shape
        a = self.assign(x).flatten(2)                     # [B, K, HW]
        a = torch.softmax(a, dim=1)
        xf = x.flatten(2)                                 # [B, d, HW]
        v = torch.einsum("bkn,bdn->bkd", a, xf)           # [B, K, d]
        v = v - a.sum(-1).unsqueeze(-1) * self.centers    # 减质心 -> 残差和
        v = torch.nn.functional.normalize(v, dim=2)       # intra-normalization
        v = torch.nn.functional.normalize(v.flatten(1), dim=1)
        return self.fc(self.dropout(v))


# 名字 -> 构造器。加新头在这里加一行即可。
# ⚠️ 必须放在所有 head 类定义之后，否则引用不到。
class CosineHead(nn.Module):
    """余弦分类头：特征和权重都 L2 归一化，logits = s · cos(θ)。

    动机来自实测（scripts/analyze_gap.py 的后续探针）：同一个模型，
    沿分类器自己的方向 d = w_a - w_b 投影，训练集 d′ = 3.1~4.3，
    测试集只有 1.1~1.8 —— 而类内散度和类中心几乎没变（shift 0.33）。
    说明问题不是特征塌了，是分类器在 2048 维里用上了「专门分开这 27 张训练图」
    的方向，其中一大块自由度来自模长。

    线性头里 logits = w·f 同时用方向和模长；这里把模长整个砍掉，
    决策只由夹角决定。温度 s 可学习（初始 16，等价于 cos 的尺度）。
    参数比线性头还少（没有 bias）。
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0,
                 scale: float = 16.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.normalize(self.dropout(x), dim=1)
        w = torch.nn.functional.normalize(self.fc.weight, dim=1)
        return self.scale * (x @ w.t())

    def cam_weight(self, feat_dim: int) -> torch.Tensor:
        return torch.nn.functional.normalize(self.fc.weight, dim=1)


HEADS = {
    "linear": LinearHead,
    "cosine": CosineHead,
    "gapstd": GAPStdHead,
    "cbp": CompactBilinearHead,
    "netvlad": NetVLADHead,
    "cap": CAPHead,
}
