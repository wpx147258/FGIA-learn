# cub-bird

在 CUB-200-2011 上做细粒度鸟类分类（200 类），先跑通一个规范的 baseline 训练/测试流程。

## 环境

- Python 3.12，PyTorch 2.14.1 + CUDA 13.0（开发机 RTX 3090 24G）
- 依赖见 `requirements.txt`。`torch` / `torchvision` 请按自己的 CUDA 版本从官方 index 装，
  其余 `pip install -r requirements.txt`。

```bash
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

## 数据

数据集放在 `data/CUB-200-2011/`（已被根 `.gitignore` 挡住，不进仓库）。
里面需要 `images.txt` / `image_class_labels.txt` / `train_test_split.txt` / `classes.txt` / `images/`。

## 怎么跑

```bash
# 训练
python train.py --config configs/baseline_config.yaml

# 冒烟测试：先跑 2 个 epoch 确认流程通
python train.py --config configs/baseline_config.yaml --set TRAIN.epochs=2

# 测试（只在最终报告时跑一次）
python test.py --checkpoint outputs/baseline_resnet50/<时间戳>/best.pth
```

命令行覆盖任意配置项，改错键名会直接报错：

```bash
python train.py --config configs/baseline_config.yaml --set TRAIN.lr=0.0005 TRAIN.batch_size=64
```

输出在 `outputs/<EXP_NAME>/<时间戳>/`：`config.yaml`（生效配置）、`log.txt`、`metrics.csv`、
`best.pth`（按 val top1 选）、`last.pth`。

## 划分方式

CUB 官方只给了 train/test（5994 / 5794）。我们从官方 train 里切 10% 做 val，
切法是**按索引等步长**（`i % 10 == 0`），不是随机采样。

原因是 CUB 的两个数据性质：image_id 按类别连续分块，且每个类的官方 train 图恰好 29~30 张。
于是「按 id 升序的 train 列表」里每个类的样本自成一整块，按索引取等步长就等价于**在每个类内部
等步长抽样**。实测两种切法（在真实数据上跑过）：

| 切法 | n_val | 每类 val 数 | 每类比例 std | 零 val 类数 |
| --- | --- | --- | --- | --- |
| **索引步长（采用）** | 599 | 2 ~ 3 | **0.0029** | **0** |
| 随机采样 | 599 | 0 ~ 8 | 0.0574 | 13 |
| 按 image_id 取模 | 614 | 0 ~ 6 | 0.0401 | 6 |

随机采样会有 13 个类在 val 里一张图都没有，那些类的 val 准确率无从计算；
按 image_id 取模也不行——类块长度在 41~60 之间不固定、起点模 10 又逐类漂移，相位一乱就散了。

最终划分：**train 5395 / val 599 / test 5794**。切分逻辑就是 `utils/data_split.py` 里的
`get_train_val_test_split()`，几行，步长默认 10（改 `VAL_STRIDE`）。

## 代码结构

```
configs/     base_config.yaml 定义全部可调项；实验配置只写 delta
model/       backbone（timm，只出特征）+ head + Classifier 组合
dataset/     CUBDataset（返回 dict）+ transforms（归一化跟随 backbone）
engine/      训练/评估循环、指标、优化器与调度器构建
utils/       配置合并、随机种子、划分解析
```

设计上留了几个扩展点（当前用不到，但改起来不用动主干）：

- **换 backbone**：改 `configs/base_config.yaml` 里的 `MODEL.name` 一个字符串即可，
  特征维度从 `backbone.num_features` 自动读取，归一化参数也从模型自身解析
- **换分类头 / 多任务**：分类头在 `model/head.py` 的 `HEADS` 里登记；
  多任务只需要改 `engine/trainer.py` 的 `forward_batch` 一个函数
  （criterion 收的是整个 batch dict，不是单个 label，属性/部位标注天然拿得到）
- **加标注字段**：`CUBDataset` 返回 dict，加一个 key 就行，下游不用动
- **消融实验**：`--set` 覆盖任意配置项，配置会落盘到 run 目录

## 结果

_待填（跑完 baseline 后补 test top-1 / top-5 / macro）_

ImageNet 预训练 ResNet50 + 224 输入 + 标准增强，CUB-200-2011 上 test top-1 的正常区间是
**70~82%**。低于 40% 基本可以确定有 bug（归一化参数错、预训练没加载、transform 把图毁了）；
高于 88% 则要怀疑测试集泄漏。

**中途检查点**：迁移学习收敛很快，epoch 5~10 时 val top-1 应该已经到 50~65%。
如果 10 个 epoch 还低于 40%，不用等跑完 100 轮，当场 debug。

## 已知问题

- `test.py` 只用 `best.pth` 评过一次，结果待补
- 当前没有多尺度/多裁剪测试（TTA），也没有 bbox 剪裁；CUB 上这两项各能涨几个点
- 划分是确定性的、不含随机性。要换随机划分，改 `utils/data_split.py`
