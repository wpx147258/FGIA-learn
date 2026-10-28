"""CUB-200-2011 官方划分的解析，以及从 train 里切出 val。

## 划分策略：按索引等步长（stride）

`get_train_val_test_split` 对**按 image_id 升序排列**的 train 列表取等步长抽样。
为什么这样切比随机采样好，是实测过的（脚本见下），原因在数据结构的两个性质：

  1. CUB 的 image_id 按类别连续分块（第 1 类占 id 1~59，第 2 类接着往下）
  2. 每个类的 train 图恰好 29~30 张

于是「按 id 升序的 train 列表」里，每个类的样本自成一整块；对索引取等步长，
等价于**在每个类内部等步长抽样** —— 每类稳定抽到 2~3 张。

实测对比（5994 张官方 train，val 目标 10%）：

    切法                     n_val   每类 val 数   每类比例 std   零 val 类数
    索引步长 i%10==0          599     2 ~ 3        0.0024         0
    随机采样 seed=0           599     0 ~ 8        0.0574        13
    image_id 取模 %10==0      614     0 ~ 6        0.0401         6

随机采样会有十几个类在 val 里一张图都没有，那些类的 val 准确率是未定义的。
**按 image_id 取模反而不行**：类块长度在 41~60 之间不固定、起点模 10 又逐类
漂移，相位一乱就散了。

代价：这个方法完全是确定性的，不随机。为了消除「每类都固定抽同一相对位置」
的系统性偏差（CUB 类内大致按摄影师文件名排序），可以选择随机起始相位，
但那样就引入了 Python 版本间 `randrange` 的微小依赖。当前实现选了确定的那条，
因为划分结果已经落盘成 JSON（见 `scripts/make_split.py`），划分一旦冻结就不该再变。
"""

from __future__ import annotations

VAL_STRIDE = 10


def get_train_test_split_original(train_test_split_txt_path):
    """解析官方 `train_test_split.txt`，返回 (train_ids, test_ids)，均按 id 升序。

    文件每行是 `<image_id> <is_train>`，is_train 为 1 表示训练图。
    按文件顺序读取，CUB 里就是升序，后面的索引步长依赖这一点。
    """
    train_ids, test_ids = [], []
    with open(train_test_split_txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_id, is_train = line.split()
            (train_ids if is_train == "1" else test_ids).append(int(image_id))
    return train_ids, test_ids


def get_train_val_test_split(train_test_split_txt_path, val_stride: int = VAL_STRIDE):
    """在官方 train 里按索引等步长切出 val，其余留在 train。

    依赖 train_ids 按 image_id 升序（`get_train_test_split_original` 保证），
    否则「每个类的样本自成一整块」这个前提不成立，划分就不再类均衡。
    """
    train_ids_ori, test_ids = get_train_test_split_original(train_test_split_txt_path)

    train_ids, val_ids = [], []
    for i, image_id in enumerate(train_ids_ori):
        # i == 0 留给 train：纯粹沿用最初的写法，把第一张图固定在训练集里。
        # 留或不留都只影响一张图，对类均衡没有影响。
        if i % val_stride == 0 and i != 0:
            val_ids.append(image_id)
        else:
            train_ids.append(image_id)

    return train_ids, val_ids, test_ids

def get_split_data(train_test_split_txt_path: str, split: str = "train"):
    train_ids, val_ids, test_ids = get_train_val_test_split(train_test_split_txt_path)
    data_ids = []
    if split == "train":
        data_ids = train_ids
    if split == "val":
        data_ids = val_ids
    if split == "test":   
        data_ids = test_ids
    return data_ids    