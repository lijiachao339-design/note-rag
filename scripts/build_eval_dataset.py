#!/usr/bin/env python
"""把 workflow 生成的候选问题编译成可评测的 ``eval/dataset.jsonl``。

为什么不直接相信模型输出，而要多这一层确定性校验：

1. **幻觉闸门**：``answer_span`` 必须能在源笔记里逐字找到（忽略空白差异）。
   编造的"原文片段"一律丢弃——这是零成本、可复现的质量控制，比再叫一个模型来判断更可靠。
2. **元问题过滤**：剔除"这篇笔记的标题是什么"这类不需要检索就能答的问题。
3. **确定性**：同一批候选文件跑两次必须得到逐字节相同的输出（编号、排序、去重都固定）。
   评测集的编号一旦漂移，历史指标就无法对比。

另外补入少量**不可回答问题**（``relevant_notes: []``）。它们的作用是暴露"检索器总能返回点什么"
这种幻觉式行为：如果评测集里全是有答案的问题，一个永远返回 Top-10 的检索器看起来也不会太差。

用法：
    uv run python scripts/build_eval_dataset.py
    uv run python scripts/build_eval_dataset.py --corpus data/corpus --out eval/dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from note_rag.bm25 import tokenize

MIN_QUESTION_CHARS = 8
MAX_QUESTION_CHARS = 200
MIN_SPAN_CHARS = 10
MAX_SPAN_CHARS = 400
VALID_DIFFICULTY = {"easy", "medium", "hard"}

# 不需要检索就能回答的元问题特征
META_PATTERNS = ("这篇笔记", "本文档", "这一篇", "标题是什么", "文件名", "文件叫什么")

# 刻意选择与本语料领域（Agent 框架 / TS 构建 / 会话持久化）相距很远的问题。
# 它们**故意**在语料里没有答案，用于度量"检索器会不会硬凑答案"。
# 脚本会用关键词稀有度自动复核，太"大众"的会被丢弃（见 keyword_rarity）。
NEGATIVE_QUESTIONS: tuple[str, ...] = (
    "Kubernetes 的 Pod 驱逐策略怎么配置软硬阈值？",
    "Rust 的借用检查器如何处理生命周期标注中的协变？",
    "C++ 里 SFINAE 和 concepts 在重载决议中的差别是什么？",
    "Android 的 WorkManager 如何保证跨进程任务只执行一次？",
    "SwiftUI 里 StateObject 和 ObservedObject 的销毁时机有何不同？",
    "Blender 的几何节点如何用场驱动顶点位移？",
    "用 CRISP-DM 做信贷违约预测，第一步该交付什么？",
    "住房贷款利息的个税专项附加扣除标准是多少？",
    "AlphaFold2 的 Evoformer 为什么用三角乘法更新？",
    "Unity DOTS 里 Burst 编译器不支持哪些 C# 特性？",
    "FPGA 上实现 AXI4-Stream 时 TREADY 反压怎么处理？",
    "InnoDB 的 next-key lock 如何避免幻读？",
    "WebGL 的 transform feedback 如何实现 GPU 粒子状态回写？",
    "跨时钟域握手时两级同步器为什么处理不了多比特信号？",
    "PCR 扩增 3kb 片段时延伸时间按什么规则设置？",
    "权责发生制与收付实现制对利润表的影响有何不同？",
    "Stockfish 的 NNUE 评估网络怎么做增量更新？",
    "为什么说 FIFO 页面置换算法会出现 Belady 异常？",
)

# 一个关键词出现在多少篇以上就算"大众" → 该负样本不可信，丢弃
NEGATIVE_MAX_DOC_FREQ = 5


@dataclass
class Stats:
    candidates_read: int = 0
    files_unreadable: int = 0
    kept: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    duplicates: int = 0
    negatives_kept: int = 0
    negatives_dropped: list[tuple[str, str]] = field(default_factory=list)


def normalize_ws(text: str) -> str:
    """压掉所有空白，用于"逐字出现"的比对（换行差异不该算改写）。"""
    return re.sub(r"\s+", "", text)


def load_candidates(directory: Path, stats: Stats) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  跳过（读取失败）{path.name}: {exc}", file=sys.stderr)
            stats.files_unreadable += 1
            continue
        questions = payload.get("questions")
        note_path = payload.get("note_path")
        if not isinstance(questions, list) or not isinstance(note_path, str):
            print(f"  跳过（结构不对）{path.name}", file=sys.stderr)
            stats.files_unreadable += 1
            continue
        for question in questions:
            if isinstance(question, dict):
                entries.append({"note_path": note_path, **question})
    return entries


def keyword_rarity(question: str, corpus: dict[str, str]) -> tuple[str, int]:
    """返回 (最稀有关键词, 该词出现的文档数)。

    关键词取 ASCII 词（长度>=4）与中文二字组，取文档频率最低的那个作为判断依据。
    文档频率越低，说明这个话题在本语料里越冷门，作为"不可回答"样本越可信。
    """
    tokens = [t for t in tokenize(question) if len(t) >= 2]
    if not tokens:
        return ("", len(corpus))
    rarest, best = "", len(corpus) + 1
    for token in set(tokens):
        df = sum(1 for text in corpus.values() if token in text)
        if df < best:
            rarest, best = token, df
    return (rarest, best)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="编译检索评测集")
    parser.add_argument("--candidates", default="eval/candidates", help="候选问题目录")
    parser.add_argument("--corpus", default="data/corpus", help="语料目录，用于校验 answer_span")
    parser.add_argument("--out", default="eval/dataset.jsonl", help="输出的数据集路径")
    parser.add_argument("--report", default="eval/dataset.report.md", help="输出的报告路径")
    args = parser.parse_args(argv)

    candidates_dir = Path(args.candidates)
    corpus_dir = Path(args.corpus)
    if not candidates_dir.is_dir():
        print(f"候选目录不存在: {candidates_dir}", file=sys.stderr)
        return 1
    if not corpus_dir.is_dir():
        print(f"语料目录不存在: {corpus_dir}", file=sys.stderr)
        return 1

    stats = Stats()
    entries = load_candidates(candidates_dir, stats)
    stats.candidates_read = len(entries)
    print(
        f"读到候选问题 {len(entries)} 条（来自 {len(list(candidates_dir.glob('*.json')))} 个文件）"
    )

    # 语料正文缓存：answer_span 校验与负样本关键词统计都要用
    corpus: dict[str, str] = {}
    for note in sorted(corpus_dir.glob("*.md")):
        try:
            corpus[note.name] = note.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"  语料读取失败 {note.name}: {exc}", file=sys.stderr)
    corpus_normalized = {name: normalize_ws(text) for name, text in corpus.items()}
    print(f"载入语料 {len(corpus)} 篇")

    kept: list[dict[str, object]] = []
    seen: set[str] = set()

    for entry in entries:
        note_path = str(entry.get("note_path", ""))
        note_name = Path(note_path.replace("\\", "/")).name
        question = str(entry.get("question", "")).strip()
        span = str(entry.get("answer_span", "")).strip()
        difficulty = str(entry.get("difficulty", "")).strip().lower()

        if note_name not in corpus:
            stats.dropped["源笔记不在语料中"] += 1
            continue
        if not (MIN_QUESTION_CHARS <= len(question) <= MAX_QUESTION_CHARS):
            stats.dropped["问题长度不合适"] += 1
            continue
        if any(pattern in question for pattern in META_PATTERNS):
            stats.dropped["元问题（无需检索即可回答）"] += 1
            continue
        if not (MIN_SPAN_CHARS <= len(span) <= MAX_SPAN_CHARS):
            stats.dropped["答案片段长度不合适"] += 1
            continue
        if normalize_ws(span) not in corpus_normalized[note_name]:
            # 这是最重要的一条：模型编造了"原文片段"
            stats.dropped["answer_span 在源笔记中找不到（疑似幻觉）"] += 1
            continue
        if difficulty not in VALID_DIFFICULTY:
            stats.dropped["difficulty 取值非法"] += 1
            continue

        key = normalize_ws(question)
        if key in seen:
            stats.duplicates += 1
            continue
        seen.add(key)

        kept.append(
            {
                "question": question,
                "note_path": note_name,
                "answer_span": span,
                "difficulty": difficulty,
            }
        )

    kept.sort(key=lambda item: (str(item["note_path"]), str(item["question"])))
    stats.kept = len(kept)

    # 负样本：先用关键词稀有度筛掉"可能其实有答案"的
    negatives: list[dict[str, object]] = []
    for question in NEGATIVE_QUESTIONS:
        token, df = keyword_rarity(question, corpus)
        if df <= NEGATIVE_MAX_DOC_FREQ:
            negatives.append({"question": question, "note_path": None, "token": token, "df": df})
        else:
            stats.negatives_dropped.append((question, f"关键词 {token!r} 出现在 {df} 篇中"))
    stats.negatives_kept = len(negatives)

    # 写数据集：正样本用 note 名，负样本 relevant_notes 为空数组
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, item in enumerate(kept, start=1):
        rows.append(
            {
                "id": f"q{index:04d}",
                "question": item["question"],
                "relevant_notes": [item["note_path"]],
                "answerable": True,
                "difficulty": item["difficulty"],
                "answer_span": item["answer_span"],
            }
        )
    for offset, item in enumerate(negatives, start=1):
        rows.append(
            {
                "id": f"n{offset:04d}",
                "question": item["question"],
                "relevant_notes": [],
                "answerable": False,
                "difficulty": "unanswerable",
                "answer_span": "",
            }
        )
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 报告
    difficulty_counts = Counter(str(item["difficulty"]) for item in kept)
    per_note = Counter(str(item["note_path"]) for item in kept)
    lines = [
        "# 评测集构建报告",
        "",
        f"- 候选问题：**{stats.candidates_read}** 条（来自 {len(list(candidates_dir.glob('*.json')))} 篇笔记）",
        f"- 通过校验：**{stats.kept}** 条",
        f"- 重复丢弃：**{stats.duplicates}** 条",
        f"- 不可回答问题：**{stats.negatives_kept}** 条（候选 {len(NEGATIVE_QUESTIONS)} 条）",
        f"- 数据集总行数：**{len(rows)}** 行 → `{out_path}`",
        "",
        "## 丢弃原因分布",
        "",
        "| 原因 | 条数 |",
        "| --- | --- |",
    ]
    for reason, count in stats.dropped.most_common():
        lines.append(f"| {reason} | {count} |")
    lines += [
        "",
        "## 难度分布（通过校验的正样本）",
        "",
        "| 难度 | 条数 |",
        "| --- | --- |",
    ]
    for level in ("easy", "medium", "hard"):
        lines.append(f"| {level} | {difficulty_counts.get(level, 0)} |")
    lines += [
        "",
        f"平均每篇笔记 {stats.kept / max(1, len(per_note)):.2f} 条问题，覆盖 {len(per_note)} 篇笔记。",
        "",
    ]
    if stats.negatives_dropped:
        lines += ["## 被丢弃的负样本（关键词太大众，可能其实有答案）", ""]
        for question, reason in stats.negatives_dropped:
            lines.append(f"- {question} —— {reason}")
        lines.append("")
    lines += [
        "## 负样本清单（已保留）",
        "",
        "| 问题 | 最稀有关键词 | 出现文档数 |",
        "| --- | --- | --- |",
    ]
    for item in negatives:
        lines.append(f"| {item['question']} | `{item['token']}` | {item['df']} |")
    lines.append("")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("\n".join(lines[:14]))
    print(f"报告已写入 {report_path}")
    print(f"数据集已写入 {out_path}（{len(rows)} 行）")
    if stats.kept < 150:
        print(
            f"⚠️ 正样本只有 {stats.kept} 条，建议再抽一批语料跑一次 workflow 后重新编译",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
