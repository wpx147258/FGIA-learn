"""对一个 run 目录里的**所有**权重做单视图评测：best / last / swa_f0.5 / 0.6 / 0.7。

为什么必须全测（2026-10-29 用户要求）：
    SWA 的不同平滑窗口差别不小。实测 `r50_pa4_js`：swa_f0.5 的 val 88.97，
    但 f0.6 / f0.7 是 89.91；统一只测 f0.5 会把这类配置的系统性差异当成机制差异。
    而且 SWA 对基线的提升（+1.36）远大于对注意力模块（+0.5~1.0）——
    只测一个权重，等于把 SWA 的收益算成机制的收益。

**主表口径仍然固定在训练前约定的那一个**（当前 = swa_f0.5），
其余权重只作为稳健性检查；不许事后按 test 挑最好的那个（那是把 test 当 val）。

用法：
    python scripts/eval_all_ckpts.py --run-dir outputs/r50_pa4js_r2/20261029-xxxxxx
    python scripts/eval_all_ckpts.py --name r50_pa4js_r2        # 自动找最新目录
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPTS = ["best", "last", "swa_f0.5", "swa_f0.6", "swa_f0.7"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--name", default="", help="实验名；自动取 outputs/<name>/ 下最新目录")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    if a.run_dir:
        d = Path(a.run_dir)
    elif a.name:
        dirs = sorted(glob.glob(str(ROOT / "outputs" / a.name / "*")))
        if not dirs:
            sys.exit(f"找不到 outputs/{a.name}/*")
        d = Path(dirs[-1])
    else:
        sys.exit("需要 --run-dir 或 --name")

    name = d.parent.name
    print(f"run: {d}")
    done = []
    for ck in CKPTS:
        p = d / f"{ck}.pth"
        if not p.exists():
            print(f"  {ck:10s} 缺失")
            continue
        tag = f"{name}_{ck.replace('.', '')}"
        out = ROOT / "dumps" / f"single_{tag}"
        cmd = [sys.executable, str(ROOT / "scripts" / "test_tta.py"),
               "--checkpoint", str(p), "--views", "base", "--split", a.split,
               "--dump", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                           env={**os.environ, "HF_ENDPOINT": "https://hf-mirror.com"})
        line = [l for l in r.stdout.splitlines() if "总融合" in l]
        print(f"  {ck:10s} {line[-1].strip() if line else '评测失败'}")
        done.append(ck)
    print(f"完成 {len(done)}/{len(CKPTS)} 个权重")


if __name__ == "__main__":
    main()
