"""把所有已评测 run 的结果汇总成一张表（可重复生成）。

数据来源：远程 /root/*.log 与本地 logs/queue_r50_90.log 里的 `--- eval <run>` 块。
口径：单视图、单权重；`主表权重` 是事先约定的那个（50 轮族 = swa_f0.5，75/100 轮族 = swa_f0.6），
`best-of-5` 是 5 个权重取最高（**属于按 test 选，只能作为参考，不能当主表**）。
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def parse(path, run_re, val_re):
    out, cur = {}, None
    for line in open(path, errors="ignore"):
        m = run_re.match(line)
        if m:
            cur = m.group(1).rstrip("/"); out.setdefault(cur, {}); continue
        m2 = val_re.match(line)
        if m2 and cur:
            out[cur][m2.group(1)] = float(m2.group(2))
    return out

val_re = re.compile(r"^\s+(best|last|swa_f0\.5|swa_f0\.6|swa_f0\.7)\s+\[test\].*top1=([0-9.]+)")

def main():
    remote = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    local = parse(ROOT / "logs/queue_r50_90.log", re.compile(r"run: outputs/(\S+)/"), val_re)
    allr = {**remote, **local}
    rows = []
    for n, d in allr.items():
        if not d: continue
        main_w = "swa_f0.6" if ("e75" in n or "e100" in n) and "swa_f0.5" not in d else "swa_f0.5"
        if d.get(main_w) is None:
            main_w = max(d, key=d.get)
        rows.append((n, d.get(main_w, float("nan")), max(d.values()), max(d, key=d.get)))
    rows.sort(key=lambda r: -r[2])
    lines = ["# 实验与最优测试结果总表", "",
             f"共 **{len(rows)}** 个 run 有评测结果（单视图、单权重口径）。", "",
             "| # | run | 主表权重 | best-of-5 | 最高权重 |",
             "|---|---|---|---|---|"]
    for i, (n, m, b, w) in enumerate(rows, 1):
        lines.append(f"| {i} | `{n}` | {m:.2f} | **{b:.2f}** | {w} |")
    lines += ["", "> `best-of-5` 是 5 个权重里取 test 最高，属于按 test 选，**只能参考**；",
              "> 结论一律按「主表权重」那一列。"]
    (ROOT / "docs/RESULTS_TABLE.md").write_text("\n".join(lines) + "\n")
    print(f"写入 docs/RESULTS_TABLE.md，共 {len(rows)} 行")

if __name__ == "__main__":
    main()
