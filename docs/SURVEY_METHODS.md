# 综述里的方法清单（Wei et al., TPAMI 2022, arXiv 2111.06119）

来源：Figure 7（2014–2020 时间线）与 Table 2（CUB / Dogs / Cars / Aircraft 对比）。
只列方法名、发表年份、骨干与 CUB 准确率；末尾按我们的约束筛一遍。

Table 2 的鸟榜数字是论文原文，训练集口径各不相同（有的用 bbox，有的用 parts），
**不能直接和我们的 89.02 比**——我们只取「ResNet-50 + 只用图像标签 + 单次前向」这一格。

---

## 一、Figure 7 时间线（按综述的三条范式）

### 1. Recognition by localization-classification subnetworks

**(a) 用检测/分割（detection or segmentation techniques）**

| 方法 | 年份 | 备注 |
|---|---|---|
| Branson et al. [64] | BMVC 2014 | 关键点 + 多块形变 |
| PB R-CNN [65] | ECCV 2014 | part-based R-CNN |
| Krause et al. [66] | CVPR 2015 | bbox 定位＋对齐 |
| Deep LAC [67] | CVPR 2015 | 定位/对齐/分类联合 |
| PS-CNN [68] | CVPR 2016 | 部件选择 |
| SPDA-CNN [69] | CVPR 2016 | 部件候选 + 检测器 |
| Zhang et al. [70] | IEEE TIP 2016 | 弱监督部件提议 + 聚类 |
| HSnet [71] | CVPR 2017 | 层次语义 |
| TSC [72] | AAAI 2017 | 三流 |
| Mask-CNN [31] | PR 2018 | 分割掩码选部件 |
| Ge et al. [73] | CVPR 2019 | 弱监督部件 |
| GCL [74] | AAAI 2020 | 通道分组 |
| FDL [75] | AAAI 2020 | 特征蒸馏式定位 |

**(b) 用深层滤波器（utilizing deep filters）**

| 方法 | 年份 | 备注 |
|---|---|---|
| Two-level attention [76] | CVPR 2015 | 谱聚类选通道当部件检测器 |
| CL [77] | CVPR 2015 | 跨层 pooling |
| NAC [78] | ICCV 2015 | 全卷积部件检测器（本文重点参考） |
| PDFS [79] | CVPR 2016 | 部件检测器逐轮更新 |
| DFL-CNN [80] | CVPR 2018 | 可学习 1×1「patch detector」+ 峰值响应 |
| S3N [81] | ICCV 2019 | 选择性稀疏采样 + 部件相关性 |
| Huang et al. [82] | CVPR 2020 | — |

**(c) 用注意力机制（leveraging attention mechanisms）**

| 方法 | 年份 | 备注 |
|---|---|---|
| RA-CNN [83] | CVPR 2017 | 循环放大，**测试要多次前向**（我们禁用） |
| MA-CNN [84] | ICCV 2017 | 多注意力图 → 多部件特征 |
| Liu et al. [39] | AAAI 2017 | 部件 + 属性，需额外标注 |
| Sun et al. [27] | ECCV 2018 | 多注意力 + 度量学习（部件相关性） |
| OPAM [85] | IEEE TIP 2018 | 目标/部件两级注意力 |
| MGE-CNN [86] | ICCV 2019 | 多粒度集成（训练期多粒度） |
| TASN [87] | CVPR 2019 | 三线性注意力采样（多部件提议） |
| PA-CNN [88] | IEEE TIP 2020 | 部件注意力 |
| Ji et al. [89] | CVPR 2020 | 注意力卷积二值树，由粗到细 |

**(d) 其它（others）**

| 方法 | 年份 | 备注 |
|---|---|---|
| STN [90] | NeurIPS 2015 | 空间变换网络 |
| BoT [91] | CVPR 2016 | 三元组模板 |
| NTS-Net [92] | ECCV 2018 | navigator–teacher–scrutinizer，**测试多次前向**（禁用） |
| M2DRL [93] | IJCV 2019 | 强化学习找区域，**多次前向**（禁用） |
| DF-GMM [94] | CVPR 2020 | 低秩基重建特征图，划分判别区域 |

### 2. Recognition by end-to-end feature encoding

**(a) 高阶特征交互（high-order feature interactions）**

| 方法 | 年份 | 备注 |
|---|---|---|
| Bilinear CNN [95] | ICCV 2015 | 二阶统计 |
| C-BCNN [96] | CVPR 2016 | 紧致双线性（Tensor Sketch） |
| KP [97] | CVPR 2017 | 核池化 |
| LRBP [98] | CVPR 2017 | 低秩双线性 |
| G²DeNet [99] | CVPR 2017 | 高斯嵌入 |
| Cai et al. [100] | ICCV 2017 | 多项式核 |
| iSQRT-COV [101] | CVPR 2018 | 协方差 + 矩阵平方根归一化 |
| DeepKSPD [102] | ECCV 2018 | 核矩阵 + 可学习 |
| HBP [103] | ECCV 2018 | 层次双线性 |
| GP [104] | ECCV 2018 | 广义池化 |
| DBTNet [105] | NeurIPS 2019 | 深度双线性变换 |
| MOMN [106] | IEEE TIP 2020 | 矩阵幂归一化 |

**(b) 专用损失（specific loss functions）**

| 方法 | 年份 | 备注 |
|---|---|---|
| MaxEnt [107] | NeurIPS 2018 | 最大化熵，防过置信 |
| PC [108] | ECCV 2018 | pairwise confusion |
| CIN [109] | AAAI 2020 | 对比通道交互 |
| Sun et al. [110] | AAAI 2020 | 梯度提升式难例损失 + 多样化模块 |
| API-Net [111] | AAAI 2020 | 成对注意力交互 |
| MC-Loss [112] | IEEE TIP 2020 | 判别性 + 多样性分量 |
| BGL [50] | CVPR 2016 | 双线性图标签（标签层级） |
| DCL [113] | CVPR 2019 | destruction + construction |
| Cross-X [114] | ICCV 2019 | 跨层注意力（layer3↔layer4） |
| PMG [115] | ECCV 2020 | 渐进多粒度拼图（训练期 4 粒度） |

### 3. Recognition with external information

| 类别 | 方法 |
|---|---|
| 网页数据（web data） | Xu et al. [150]、Krause et al. [151]、Niu et al. [152]、Yang et al. [153]、Xu et al. [154]、MetaFGNet [155]、Zhang et al. [156] |
| 多模态（multi-modality） | CVL [59]、T-CNN [158]、KERL [159]、PMA [160] |
| 人在回路（human in the loop） | Dong et al. [157]、Cui et al. [166] |

（这一类我们用不了：约束是不引入外部数据。）

---

## 二、Table 2 原表（只摘 CUB 一列）

| 方法 | 发表 | 骨干 | 输入 | CUB |
|---|---|---|---|---|
| Branson [64] | BMVC 2014 | CaffeNet | 224 | 75.7 |
| PB R-CNN [65] | ECCV 2014 | AlexNet | 224 | 76.4 |
| Krause [66] | CVPR 2015 | CaffeNet | 224 | 82.0 |
| Deep LAC [67] | CVPR 2015 | AlexNet | 227 | 80.3 |
| PS-CNN [68] | CVPR 2016 | CaffeNet | 227 | 76.6 |
| SPDA-CNN [69] | CVPR 2016 | CaffeNet | 800 | 81.0 |
| SPDA-CNN [69] | CVPR 2016 | VGG-16 | 800 | 85.1 |
| Zhang [70] | TIP 2016 | AlexNet | 224 | 78.9 |
| HSnet [71] | CVPR 2017 | GoogLeNet | 224 | 87.5 |
| TSC [72] | AAAI 2017 | VGG-16 | — | 84.7 |
| Mask-CNN [31] | PR 2018 | VGG-16 | 448 | 85.7 |
| Ge [73] | CVPR 2019 | GoogLeNet+BN | 800 | **90.3**（⚠️ **不是**关键点监督：原文明确 "without any parts annotations"，用 Mask R-CNN + CRF 弱监督；不能拿来支持「关键点能到 90」） |
| GCL [74] | AAAI 2020 | ResNet-50 | 448 | 88.3 |
| FDL [75] | AAAI 2020 | ResNet-50 | 448 | 88.6 |
| Two-level atten. [76] | CVPR 2015 | VGG-16 | — | 77.9 |
| CL [77] | CVPR 2015 | AlexNet | 448 | 73.5 |
| NAC [78] | ICCV 2015 | VGG-16 | — | 81.0 |
| PDFS [79] | CVPR 2016 | VGG-16 | — | 84.5 |
| DFL-CNN [80] | CVPR 2018 | VGG-16 | 448 | 86.7 |
| S3N [81] | ICCV 2019 | ResNet-50 | 448 | 88.5 |
| Huang [82] | CVPR 2020 | ResNet-101 | 448 | 87.3 |
| RA-CNN [83] | CVPR 2017 | VGG-19 | 448 | 85.3 |
| MA-CNN [84] | ICCV 2017 | VGG-19 | 448 | 86.5 |
| Liu [39] | AAAI 2017 | ResNet-50 | 448 | 85.4（需 Parts+Attr） |
| Sun [27] | ECCV 2018 | ResNet-50 | 448 | 86.5 |
| OPAM [85] | TIP 2018 | VGG-16 | — | 85.8 |
| MGE-CNN [86] | ICCV 2019 | ResNet-50 | 448 | 88.5 |
| TASN [87] | CVPR 2019 | ResNet-50 | 224 | 87.9 |
| PA-CNN [88] | TIP 2020 | VGG-19 | 448 | 87.8 |
| Ji [89] | CVPR 2020 | ResNet-50 | 448 | 88.1 |
| STN [90] | NeurIPS 2015 | GoogLeNet+BN | 448 | 84.1 |
| NTS-Net [92] | ECCV 2018 | ResNet-50 | 448 | 87.5 |
| M2DRL [93] | IJCV 2019 | VGG-16 | 448 | 87.2 |
| DF-GMM [94] | CVPR 2020 | ResNet-50 | 448 | 88.8 |
| Bilinear CNN [95] | ICCV 2015 | VGG-16+M | 448 | 84.1 |
| C-BCNN [96] | CVPR 2016 | VGG-16 | 448 | 84.3 |
| KP [97] | CVPR 2017 | VGG-16 | 224 | 86.2 |
| LRBP [98] | CVPR 2017 | VGG-16 | 224 | 84.2 |
| G²DeNet [99] | CVPR 2017 | VGG-16 | 200 | 87.1 |
| Cai [100] | ICCV 2017 | VGG-16 | 448 | 85.3 |
| iSQRT-COV [101] | CVPR 2018 | ResNet-101 | 224 | 88.7 |
| DeepKSPD [102] | ECCV 2018 | VGG-16 | 448 | 86.5 |
| HBP [103] | ECCV 2018 | VGG-16 | 448 | 87.1 |
| GP [104] | ECCV 2018 | VGG-16 | 448 | 85.8 |
| DBTNet-50 [105] | NeurIPS 2019 | VGG-16 | 448 | 87.5 |
| MOMN [106] | TIP 2020 | VGG-16 | 448 | 87.3 |
| MaxEnt [107] | NeurIPS 2018 | VGG-16 | 224 | 77.0 |
| MaxEnt [107] | NeurIPS 2018 | Bilinear CNN | 224 | 85.3 |
| PC [108] | ECCV 2018 | Bilinear CNN | 224 | 85.6 |
| Sun [27] | ECCV 2018 | ResNet-50 | 448 | 86.5 |
| CIN [109] | AAAI 2020 | ResNet-101 | 448 | 88.1 |
| Sun [110] | AAAI 2020 | ResNet-50 | 448 | 88.6 |
| API-Net [111] | AAAI 2020 | ResNet-50 | 448 | 87.7 |
| API-Net [111] | AAAI 2020 | DenseNet-161 | 448 | **90.0** |
| MC-Loss [112] | TIP 2020 | Bilinear CNN | 448 | 86.4 |
| BGL [50] | CVPR 2016 | VGG-16 | 224 | 75.9 |
| BGL [50]（用 bbox） | CVPR 2016 | VGG-16 | 224 | 80.4 |
| DCL [113] | CVPR 2019 | ResNet-50 | 448 | 87.8 |
| Cross-X [114] | ICCV 2019 | ResNet-50 | 448 | 87.7 |
| **PMG [115]** | ECCV 2020 | **ResNet-50** | 448 | **89.6** |

---

## 三、按我们的约束筛完剩什么

约束：ResNet-50、只用图像标签、**测试只前向一次**、不蒸馏、不集成、不并入验证集、无外部数据。

**直接出局**
- 用 bbox / parts / 属性的：Branson、PB R-CNN、Krause、Deep LAC、PS/SPDA-CNN、Zhang、
  HSnet、TSC、Mask-CNN、Ge、Liu[39]、STN 之外的大部分早期工作。
- 测试多次前向：RA-CNN、NTS-Net、M2DRL（还有 MA-CNN 的多尺度变体）。
- 外部数据/多模态：第 3 类全部。
- 我们已实测无效：双线性/紧致双线性、GeM、GAP+std、余弦头、head_norm、
  API-Net 成对判别、硬负样本 CE、熵正则、SE（末端）、TokenLearner、attn_pool（末端）、
  PMG-50ep（89.25）、深度监督、背景掩码。

**还剩、且值得排队的**
1. **DF-GMM [94]**（R50/448/88.8）：用低秩基重建特征图、按重建误差划分判别区域。
   单次前向、弱监督、无额外标注。思路与「在 layer4 上做注意力」不同——它改的是
   *特征图本身的分解方式*。
2. **S3N [81]**（R50/448/88.5）：选择性稀疏采样 + 部件相关性。单次前向。
3. **NAC [78] / DFL-CNN [80]**：可学习部件检测器 + 峰值响应图。单次前向、参数少。
   我们只做过 TokenLearner（软空间掩码），没做过「峰值响应 + 部件级池化」。
4. **Cross-X [114]**（R50/448/87.7）：layer3↔layer4 跨层注意力。我们做过 concat 探针
   （85.17，低于单用 layer4），但没做过跨层注意力融合。
5. **MGE-CNN [86]**（R50/448/88.5）：训练期多粒度，测试单次前向。和 PMG 同族。
6. **PMG [115] 训满**：论文 89.6 是 300 epoch、ImageNet 权重；我们 50 epoch、SWSL 权重
   只有 89.25（还关掉了 CAM 分支）。这是 Table 2 里 R50 单前向的最高分，值得用长训练重跑。

**判断**：Table 2 里 R50 + 单前向的天花板就是 89.6（PMG），而且没有任何一个注意力方法
在 R50 上超过 89。我们要破 90，等于要超过这张表。可行路径只有两条：
(a) 把某个机制做对（不是照抄，而是补上它没做对的那一环）；
(b) 用比论文更强的训练配方（我们已有 CAM 分支 + 一致性 + SWSL，基线 89.02 已高于表里
    绝大多数 R50 方法）再叠一个真正有增量的结构改动。

这也解释了为什么「末端加注意力」全军覆没：Table 2 的注意力方法基本都建在 VGG 上，
且多数靠多次前向或额外标注；真正在 R50 单前向上有效的，是**改特征图分辨率/粒度的**
那些（FDL、DF-GMM、S3N、PMG），而不是「在 14×14 上再做一次加权」。

---

## 四、补记：两条零训练诊断（2026-10-27）

排完上面这张表后先做了两个不训练的实验，用来决定押哪条路。

**1. 测试时放大还有多少空间**（`scripts/probe_zoom.py`，基线单视图 88.78）
按模型自己的 CAM 框裁出判别区、放大到 448 再前向一次：

| 视图 | top1 |
|---|---|
| global（整图） | 88.78 |
| crop（CAM 框放大） | 84.67 |
| 两路概率平均 | 88.56 |
| oracle（任一路对就算对） | **90.01** |

crop 单独救回 71 张、global 独有 309 张，两路都错 579 张。「测试时放大」的理论上限
只有 +1.2，实际融合还是负的。综述里 RA-CNN / NTS-Net / M2DRL 那一族（靠测试时多看
几处涨点）在这套设定下没有空间。

**2. 注意力池化到底是不是「加料」**（`scripts/probe_pretrained.py`，冻结特征 + 线性探针，448，val）

| 权重 | 池化 | val top1 |
|---|---|---|
| SWSL R50 | GAP | **74.46** |
| CLIP RN50 | GAP | 37.73 |
| CLIP RN50 | 注意力池化（AttentionPool2d） | **59.93** |

同一套 CLIP 权重，只换池化头，差 22.2 个点。三个结论：
- 注意力池化不是可有可无的加料，它决定特征可不可分；
- 但它是**预训练阶段长在权重里**的。SWSL R50 末端从零加一个，等于让 5400 张图去学
  预训练该学好的部分——这正是此前四个注意力变体（AttnPool 196/49、d512、TokenLearner）
  全部无效的原因；
- CLIP RN50 的 GAP 特征本身很弱（37.73），说明「池化方式必须与预训练方式匹配」。

---

## 五、[70]–[76] 逐篇拆解：部件提取 × 自注意力

这七篇的共同骨架是「先拿部件，再让部件之间交换信息」，差别在**部件怎么来**和
**怎么融合**。把融合器换成自注意力是自然的（[74] 本来就用图传播，[71]/[73] 用 LSTM）。

| 编号 | 论文 | 部件怎么来（弱监督） | 融合器 | 测试成本 | CUB |
|---|---|---|---|---|---|
| [76] | Xiao et al., CVPR'15 两级注意力 | 深度滤波器做谱聚类 → 滤波器组当部件检测器 | 各部件特征拼接 | 单次前向 | 77.9 (VGG16) |
| [70] | Zhang et al., TIP'16 | 物体提议 + 空间金字塔 → 提议聚类成部件原型 | 选判别原型后拼接 | 提议需多次裁剪 | 78.9 (AlexNet) |
| [72] | He & Peng, AAAI'17 | 部件选择 + 部件间空间约束（相对位置） | 空间约束下的选择 | 单次前向 | 84.7 (VGG16) |
| [73] | Ge et al., CVPR'19 | Mask R-CNN + CRF 先抠物体，再自底向上找互补部件 | 双向 LSTM | 多阶段，依赖外部模型 | 90.3 (GoogLeNet+BN, 800)；**训练不用 parts 标注** |
| [74] | Wang et al., AAAI'20 (GCL) | 判别区域组 | **十字交叉图传播**（同行/同列区域互看）+ 相关性强化 | 单次前向 | 88.3 (R50, 448) |
| [75] | Liu et al., AAAI'20 (FDL) | RPN 提区域，用「提议-预测匹配度」过滤 | 区域特征从物体特征**蒸馏** | 单次前向（带 RPN） | 88.6 (R50, 448) |
| [71] | Lam et al., CVPR'17 (HSnet) | LSTM 串行搜索有用部件 | LSTM | **多步，测试多次前向** | — |

**能直接拿过来、且符合我们约束的只有三篇**：[76]（部件检测器）、[72]（空间约束）、
[74]（部件间图传播）。[73] 依赖 Mask R-CNN/CRF，[70] 需要提议裁剪，[71] 测试多步，
[75] 的蒸馏虽是自己教自己，但和本项目「不用蒸馏」的约定冲突，先不碰。

### 我们的设计：PartAttention（已实现，见 model/part_attn.py）

把 [76] 的「滤波器组当部件检测器」和 [74] 的「部件间传播」合起来，融合器换成自注意力：

1. **部件检测器**：在 layer4 特征图 [B,2048,14,14] 上做 1×1 卷积，得 K 张响应图，
   softmax 到空间维 → 每个部件的空间分布（[76] 的可学习版本）；
2. **部件描述子**：z_k = Σ_p a_k(p)·f(p)，得到 K 个 2048 维部件特征；
3. **部件间自注意力**：K 个 token（降维到 256 + 部件身份嵌入）过一层 pre-LN Transformer，
   每个部件都能看到其余 K−1 个（[74] 图传播的稠密版本，也是 [71]/[73] LSTM 的并行替代）；
4. **融合**：部件 token 平均 → 线性回 2048（零初始化）→ 加回特征图，再 GAP 分类。

参数 1.615M（K=16），控制臂（`pa_depth=0`，同样的参数预算但不做部件间混合）1.614M。
输出零初始化，训练第 0 步与基线逐位等价。模块挂在 layer4 的 forward hook 上，
所以 eval、CAM 裁剪分支、y=None 早退路径、BN 重估全都自动经过它。

诊断量：部件分布熵（越小＝越像在看某个具体位置）、部件间重叠（越大＝部件塌缩成同一处）、
输出残差相对 GAP 的幅度。这三个量决定「没涨」是因为部件没分化，还是因为部件分化了但没用。

### 一个要提前说清的预期

[76]/[72] 是在 VGG16、单阶段、无强预训练的配方下拿的 77–85 分，我们的基线（SWSL +
CAM 裁剪 + 一致性）已经是 89.4。所以**照抄它们的绝对分数没有意义**，要看的只有两件事：
部件注意力有没有让部件的空间分布分化（熵/重叠），以及分化之后单视图有没有超过基线。

---

## 六、精读原文后的结论（2026-10-27，两路独立精读）

读了 [76] Xiao CVPR'15、[72] He & Peng AAAI'17、[70] Zhang TIP'16（预印本）与
GCL（AAAI'20）、FDL（AAAI'20）、MA-CNN（ICCV'17）的原文。最重要的一条事实：

> **这六篇没有一篇是单次前向的。** 它们的增益主要来自「把候选区域裁出来、重新过一遍
> 骨干」，而不是来自部件之间的交互。GCL 的消融最能说明问题：不加任何交互，仅
> 「多尺度 default patches + 判别分数」就 +1.7（84.5→86.2），完整的十字图传播再 +0.5，
> CFS 再 +0.6。也就是说交互只占增益的三分之一，其余三之二来自多裁。

这与我们自己的两条实测完全一致：测试时 CAM 放大再前向一次的上限只有 90.01（基线 88.78），
而冻结特征的线性探针显示注意力池化头比 GAP 高 22 点——但那是预训练长出来的。

### 能搬到单次前向里的（已照原文实现）

| 来源 | 设计 | 我们怎么用 |
|---|---|---|
| MA-CNN Eq.4–5 | 通道分组 + sigmoid 掩码池化 | K 个 1×1 部件检测器，sigmoid 掩码，归一化后加权池化成部件描述子 |
| MA-CNN Eq.6 | 逐部件分类损失 | 每个部件过同一个分类头算 CE（`pa_aux_w`） |
| MA-CNN Eq.8–9 | 紧凑项 Dis + 散度项 Div（mrg=0.02，λ=2） | 模块内算 `Dis + 2·Div`，外层乘 `pa_div_w`；Div 改成就位置取平均（原文掩码稀疏，求和尺度不适用） |
| GCL Eq.6 | 零初始化残差门 α≈0 | 输出投影零初始化，初始等价基线 |
| GCL CGP | 十字/轴向聚合 | 换成 K 个部件 token 的一层稠密自注意力（用户提的那一步） |
| Xiao Eq.1 | 最终分数 = object + α·part（α=0.5） | 零初始化残差自己学比例，不写死 |

### 明确不搬的

- Xiao 的 selective search + FilterNet(0.9) + 每图约 40 patch、10 视图测试；
- He 的 saliency 反传 + co-seg + 组合优化选部件（不可微、多次前向）；
- Zhang 的每 proposal 一次 CNN + 26 万维 FV + MI 选簇 + SVM（后处理）；
- GCL 的 test-time 多 patch 分类与概率集成、FDL 的 RPN + 三分支集成、MA-CNN 的
  96×96 裁剪 + part-CNN 二次前向。

### 一个必须写进报告的前提

在「一张图、前向一次」的约束下，部件机制**不可能**产生新信息——它只能在已经算好的
14×14 特征图上重新加权。文献里部件方法的增益，主要来自把局部放大后**重新编码**，
那需要额外的前向。我们已经有等价的东西：CAM 裁剪分支 + 一致性（训练期做，测试不用）。
所以这条线的合理预期是「小幅度正则收益」，不是 +2。

---

## 七、部件注意力的四个变体为何都塌回同一张图（2026-10-28 实测）

| 变体 | 分配规则 | 初始化 overlap | 训练中 overlap |
|---|---|---|---|
| pa（sigmoid 掩码 + Div/Dis） | 每个部件独立 [0,1] 掩码 | ~1.00 | 1.00 |
| pa2（跨部件竞争 softmax, tau=1） | 每格在 K 个部件间归一化 | 0.62 | 0.98（ep9） |
| pa3（同上, tau=0.1 硬分配） | 近乎 one-hot 归属 | 0.056 | 0.998（ep42） |
| slots（Slot Attention, 3 次迭代） | 跨 slot softmax + 按 token 归一化 | 0.996（均匀） | — |

共同点：初始化时可以分开（pa3 的 0.056 是真的分开了），但训练把它们推回同一个解。
原因是目标函数本身：逐部件 CE 用「均匀池化 + 同一个分类器」就能满足，散度/均衡项
在 10^-2 量级，而主 CE 在 ~5。**没有任何一项要求部件必须不同。**

结构性例外是通道分组（pa4）：即使 K 个掩码完全相同，各部件描述子也来自不相交的
通道子集，K 个 token 必然不同。但这也意味着 K 个部件均值拼起来 ≈ GAP，
信息量与基线相同，最多是正则收益。

真正的信息增量只能来自「把区域裁出来放大后重新编码」——但这条路的上限实验显示：
CAM 框单区域 oracle 90.01（融合实测 −0.59），GT 头/身体双区域 oracle 91.85。
差在区域语义：CAM 框与全局预测高度相关，GT 部件才提供互补证据。
所以下一步的关键不是融合器，而是**部件区域能否被发现得足够互补**（NTS-Net 的
navigator + 排序损失、MA-CNN 的通道分组、GCL 的区域组）。
