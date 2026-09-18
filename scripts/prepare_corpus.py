#!/usr/bin/env python
"""把散落各处的 markdown 汇总成一份**可复现**的检索语料。

为什么需要它：检索指标只有在语料固定时才有意义。如果语料是"我机器上随便哪个目录"，
你的 recall@5 就无法被别人复现，也就无法被验证。这个脚本把语料固化下来并输出指纹。

用法示例：

    # 把某个目录下的 markdown 汇总成语料
    uv run python scripts/prepare_corpus.py \
        --src "C:/path/to/your/vault" \
        --out data/corpus

    # 多加一路语料并限制总量（脚本会按内容 sha1 去重、再均匀抽样）
    uv run python scripts/prepare_corpus.py \
        --src data/downloaded --src "C:/path/to/another/notes" \
        --out data/corpus --limit 2500

    # 只看统计，不写文件
    uv run python scripts/prepare_corpus.py --src <dir> --out data/corpus --dry-run

输出：
    <out>/*.md            归一化后的语料（扁平命名，避免路径冲突）
    <out>/MANIFEST.jsonl  每篇的 源路径 / sha1 / 字节数 / mtime
    <out>/FINGERPRINT.txt 语料指纹（把所有 sha1 排序后再哈希）——写进 docs/corpus.md

注意：data/ 已在 .gitignore 中，语料**不要**提交进仓库（尤其是别人的私有仓库内容）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SKIP_DIRS = {".git", ".obsidian", ".trash", "node_modules", "__pycache__", ".venv", "dist", "build"}
SAFE_NAME = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff._-]+")


@dataclass(slots=True, frozen=True)
class Candidate:
    src: Path
    rel: str
    size: int
    sha1: str

    @property
    def out_name(self) -> str:
        """扁平但可追溯的文件名：把源目录层级压进名字，避免同名覆盖。"""
        stem = self.rel.rsplit(".", 1)[0]
        parts = [SAFE_NAME.sub("_", p).strip("_") for p in Path(stem).parts if p not in ("", ".")]
        parts = [p for p in parts if p]
        joined = "__".join(parts[-4:])  # 只保留最后 4 层，避免文件名过长
        joined = joined[:150] or "note"
        return f"{joined}__{self.sha1[:8]}.md"


def iter_markdown(roots: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"语料源不存在: {root}")
        if root.is_file():
            if root.suffix.lower() in {".md", ".markdown"}:
                found.append(root)
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in {".md", ".markdown"} or not path.is_file():
                continue
            rel_parts = path.relative_to(root).parts
            if any(part in SKIP_DIRS for part in rel_parts):
                continue
            found.append(path)
    return sorted(set(found))


def sha1_of(data: bytes) -> str:
    return hashlib.sha1(data, usedforsecurity=False).hexdigest()


def collect(roots: Iterable[Path], *, min_bytes: int, limit: int | None) -> list[Candidate]:
    candidates: list[Candidate] = []
    for path in iter_markdown(roots):
        try:
            data = path.read_bytes()
        except OSError as exc:
            print(f"  跳过（读取失败）: {path}: {exc}", file=sys.stderr)
            continue
        if len(data) < min_bytes:
            continue
        # 源路径的展示形式：尽量相对，便于复现
        try:
            rel = path.resolve().relative_to(Path.cwd()).as_posix()
        except ValueError:
            rel = path.as_posix()
        candidates.append(Candidate(src=path, rel=rel, size=len(data), sha1=sha1_of(data)))

    # 内容去重：同一篇被复制多次时只保留一份
    seen: set[str] = set()
    unique: list[Candidate] = []
    for item in candidates:
        if item.sha1 in seen:
            continue
        seen.add(item.sha1)
        unique.append(item)

    unique.sort(key=lambda c: c.rel)
    # 均匀抽样，避免抽样结果被目录顺序偏置
    if limit is not None and limit > 0 and len(unique) > limit:
        step = len(unique) / limit
        unique = [unique[int(i * step)] for i in range(limit)]
    return unique


def fingerprint(items: Iterable[Candidate]) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    for sha in sorted(item.sha1 for item in items):
        digest.update(sha.encode("ascii"))
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="汇总并固化检索语料")
    parser.add_argument("--src", action="append", required=True, help="语料源目录或文件（可重复）")
    parser.add_argument("--out", default="data/corpus", help="输出目录")
    parser.add_argument("--limit", type=int, default=0, help="最多保留多少篇（0 = 不限）")
    parser.add_argument("--min-bytes", type=int, default=200, help="小于该字节数的文件视为噪声跳过")
    parser.add_argument("--clean", action="store_true", help="写入前清空输出目录")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    args = parser.parse_args(argv)

    roots = [Path(p) for p in args.src]
    out = Path(args.out)

    print(f"扫描 {len(roots)} 个源 ...")
    items = collect(roots, min_bytes=args.min_bytes, limit=args.limit or None)
    if not items:
        print("没有找到任何 markdown（检查 --src 与 --min-bytes）", file=sys.stderr)
        return 1

    total_bytes = sum(item.size for item in items)
    print(f"入选 {len(items)} 篇，共 {total_bytes / 1024 / 1024:.2f} MB")
    print(f"语料指纹: {fingerprint(items)}")

    if args.dry_run:
        for item in items[:5]:
            print(f"  {item.rel}  ({item.size} B)")
        print("dry-run：未写入任何文件")
        return 0

    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = out / "MANIFEST.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for item in items:
            target = out / item.out_name
            shutil.copyfile(item.src, target)
            manifest.write(
                json.dumps(
                    {
                        "out": target.name,
                        "src": item.rel,
                        "sha1": item.sha1,
                        "bytes": item.size,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    (out / "FINGERPRINT.txt").write_text(
        f"files={len(items)}\nbytes={total_bytes}\nfingerprint={fingerprint(items)}\n",
        encoding="utf-8",
    )

    print(f"已写入 {out}（{len(items)} 篇 + MANIFEST.jsonl + FINGERPRINT.txt）")
    print("下一步：把 NOTE_RAG_VAULT_PATH 指向该目录，再跑 note-rag ingest --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
