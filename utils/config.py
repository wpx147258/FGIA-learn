"""配置加载。

约定：
  * `configs/base_config.yaml` 是**完整的一份配置**（所有键都在这里定义）
  * 实验配置（如 `configs/baseline_config.yaml`）只写与 base 不同的 delta
  * 命令行 `--set TRAIN.lr=0.0005` 做最后覆盖

合并顺序：base <- 实验配置 <- --set，后者覆盖前者。

为什么不用 Hydra：单卡小项目用不上 config group 和自动 sweep，
深合并 + --set 一共几十行，够用且没有额外的目录约定要记。
真要批量扫参了再换，那时 `--set` 的用法可以原样保留。
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

# 项目根（本文件是 <root>/utils/config.py）。
# 所有 config 里的相对路径都锚到这里，而不是 cwd —— 这样无论你从仓库根
# 还是从项目目录里跑 `python train.py`，路径都能解析对。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_CONFIG = PROJECT_ROOT / "configs" / "base_config.yaml"


def resolve_path(p) -> Path:
    """把 config 里的路径解析成绝对路径（相对项目根）。"""
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并：两边都是 dict 就逐层往下，否则 override 直接替换。"""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def parse_overrides(pairs) -> dict:
    """把 ['TRAIN.lr=0.0005', 'MODEL.kwargs.pretrained=false'] 解析成嵌套 dict。

    值的类型强转交给 yaml.safe_load —— 所以 0.0005 是 float、true 是 bool、
    [224, 224] 是 list，不用自己写一堆 isinstance 判断。
    """
    out: dict = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"--set 参数必须形如 KEY=VALUE，收到: {item!r}")

        key, _, raw = item.partition("=")  # 只按第一个 = 切，值里可以再出现 =
        if raw.strip() == "":
            raise ValueError(
                f"--set {key}= 的值是空的。如果确实想设成 None，请显式写 {key}=null"
            )

        node = out
        parts = key.split(".")
        for seg in parts[:-1]:
            node = node.setdefault(seg, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def _check_keys_exist(cfg: dict, override: dict, prefix: str = "") -> None:
    """--set 写了配置里不存在的键时报错。

    不做这个检查的话，`--set TRAIN.lrr=0.1` 这种拼写错误会静默失效，
    你会跑完一整个实验才发现自己改的参数根本没生效。这是消融实验里
    最高频的自伤，值得多这十行。
    """
    for key, val in override.items():
        path = f"{prefix}{key}"
        if key not in cfg:
            raise KeyError(f"--set 覆盖了配置里不存在的键：{path}")
        if isinstance(val, dict):
            if not isinstance(cfg[key], dict):
                raise KeyError(f"--set {path} 用了嵌套写法，但配置里的 {path} 不是 dict")
            _check_keys_exist(cfg[key], val, prefix=f"{path}.")


def load_config(config_paths, overrides=()) -> dict:
    """加载并合并配置。

    config_paths: 按顺序合并，**base 放第一个**，实验配置在后（覆盖前者）
    overrides:    命令行 --set 的原始字符串列表
    """
    cfg: dict = {}
    for path in config_paths:
        with open(resolve_path(path)) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})

    ov = parse_overrides(overrides)
    _check_keys_exist(cfg, ov)
    return deep_merge(cfg, ov)


def dump_config(cfg: dict, path) -> None:
    """把生效的配置落盘到 run 目录，保证这次实验可复现。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
