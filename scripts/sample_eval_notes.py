#!/usr/bin/env python
"""确定性地为评测集抽样目标笔记，并记录抽样规则。

为什么需要它：review-001 的 S8 指出"60 篇目标笔记 / 1947 篇语料 = 3.08%，
抽样规则没有任何记录（`eval/paths.json` 是一个裸的路径数组，没有说明是怎么选的）"。
"覆盖 60 篇 / 1947 篇，按 X 规则抽取"是一句必须能回答的话。这个脚本就是那条规则：

* **确定性**：按文件名排序后等步长抽样（stride = max(1, len(pool) // count)）。
  同样的输入永远得到同样的输出——评测集的样本一旦漂移，历史指标就无法对比。
* **可排除**：`--exclude` 传入已有清单，避免同一篇笔记被抽两次
  （重复会污染"按笔记聚簇"的 bootstrap：笔记数会被虚报）。
* **记录**：`--manifest` 输出抽样元信息（池大小、步长、排除数、语料指纹），供报告引用。

用法：
    # 首批：45 篇英文 + 15 篇中文
    uv run python scripts/sample_eval_notes.py --lang en --count 45 --out eval/paths-en.json
    uv run python scripts/sample_eval_notes.py --lang zh --count 15 --out eval/paths-zh-1.json

    # 追加中文（排除首批用过的）
    uv run python scripts/sample_eval_notes.py --lang zh --count 30 \
        --exclude eval/paths-zh-1.json --exclude eval/paths-en.json --out eval/paths-zh-2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ZH_MARKER = ".zh__"


def read_json(path: Path) -> object:
    """读 JSON，容忍 PowerShell 写出的 UTF-8 BOM（`utf-8-sig` 同时吃 BOM 与无 BOM）。"""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def used_names(paths: list[Path]) -> set[str]:
    names: set[str] = set()
    for path in paths:
        if not path.exists():
            print(f"  排除清单不存在，忽略: {path}", file=sys.stderr)
            continue
        payload = read_json(path)
        if isinstance(payload, list):
            names.update(Path(str(item).replace("\\", "/")).name for item in payload)
    return names


def corpus_fingerprint(files: list[Path]) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    for path in sorted(files, key=lambda p: p.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="确定性抽样评测目标笔记")
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--lang", choices=["en", "zh", "any"], default="any")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--min-bytes", type=int, default=1024)
    parser.add_argument("--exclude", action="append", default=[], help="已用清单（可重复）")
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", default="", help="抽样元信息输出路径（可选）")
    args = parser.parse_args(argv)

    corpus = Path(args.corpus)
    if not corpus.is_dir():
        print(f"语料目录不存在: {corpus}", file=sys.stderr)
        return 1

    everything = [p for p in corpus.glob("*.md") if p.is_file()]
    pool = [
        p
        for p in everything
        if p.stat().st_size >= args.min_bytes
        and (args.lang == "any" or ((ZH_MARKER in p.name) == (args.lang == "zh")))
    ]
    pool.sort(key=lambda p: p.name)

    excluded = used_names([Path(p) for p in args.exclude])
    available = [p for p in pool if p.name not in excluded]
    if len(available) < args.count:
        print(
            f"可选池不足：需要 {args.count} 篇，排除后只剩 {len(available)} 篇"
            f"（池 {len(pool)}，已排除 {len(excluded)}）",
            file=sys.stderr,
        )
        return 1

    stride = max(1, len(available) // args.count)
    picked: list[Path] = []
    for index in range(0, len(available), stride):
        picked.append(available[index])
        if len(picked) == args.count:
            break

    paths = [p.as_posix() for p in picked]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(paths, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",  # 不带 BOM：下游 python 直接可读
    )

    manifest = {
        "out": out.as_posix(),
        "lang": args.lang,
        "count": len(picked),
        "corpus_files": len(everything),
        "pool_after_lang_and_size_filter": len(pool),
        "excluded_names": len(excluded),
        "available": len(available),
        "stride": stride,
        "min_bytes": args.min_bytes,
        "corpus_fingerprint_names_and_sizes": corpus_fingerprint(everything),
        "first_three": [Path(p).name for p in paths[:3]],
    }
    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    print(
        f"语料 {len(everything)} 篇 → 过滤后池 {len(pool)} → 排除已用 {len(excluded)} "
        f"→ 可选 {len(available)}，步长 {stride}，抽中 **{len(picked)}** 篇"
    )
    print(f"写入 {out}")
    for path in paths[:3]:
        print(f"  {Path(path).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
