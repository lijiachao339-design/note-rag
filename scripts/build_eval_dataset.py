#!/usr/bin/env python
"""把 workflow 生成的候选问题编译成可评测的 ``eval/dataset.jsonl``。

为什么不直接相信模型输出，而要多这一层确定性校验：

1. **幻觉闸门**：``answer_span`` 必须能在源笔记里逐字找到（忽略空白差异）。
   编造的"原文片段"一律丢弃——这是零成本、可复现的质量控制，比再叫一个模型来判断更可靠。
2. **元问题过滤**：剔除"这篇笔记的标题是什么"这类不需要检索就能答的问题。
3. **确定性**：同一批候选文件跑两次必须得到逐字节相同的输出（编号、排序、去重都固定）。
   评测集的编号一旦漂移，历史指标就无法对比。
4. **词法泄漏标记**（``leak``）：问题是模型读着目标笔记写的，会把该笔记的稀有标识符逐字抄进去。
   这种题 BM25 不用检索就能精确匹配，实测 df≤2 那一组 ``keyword`` recall@5 高达 1.000。
   标记而**不删除**——被标记的样本本身是好材料，只是不能混进总均值里当"检索质量"引用。
5. **分级相关**：语料是严格的双语镜像，每篇笔记都有一篇内容等价的译文孪生。
   孪生同样能回答该问题，把它判为不相关是错的；但它也不是问题所出自的那一篇。
   所以原文记 grade 2、孪生记 grade 1（``metrics.py`` 原生支持分级相关）。

另外补入少量**不可回答问题**（``relevant_notes: []``）。它们的作用是暴露"检索器总能返回点什么"
这种幻觉式行为：如果评测集里全是有答案的问题，一个永远返回 Top-10 的检索器看起来也不会太差。

负样本的复核闸门用的是**最佳笔记覆盖率**：全语料里覆盖该问题内容词最多的那一篇，覆盖了几成。
覆盖率越低，说明没有任何一篇笔记谈论这个话题，该问题作为"不可回答"越可信。

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
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from note_rag.bm25 import tokenize
from note_rag.notes import iter_vault_files

MIN_QUESTION_CHARS = 8
MAX_QUESTION_CHARS = 200
MIN_SPAN_CHARS = 10
MAX_SPAN_CHARS = 400
VALID_DIFFICULTY = {"easy", "medium", "hard"}

# 相关等级：问题所出自的那一篇 = 2，它的译文孪生 = 1。
GRADE_SOURCE = 2
GRADE_TRANSLATION = 1

# 词法泄漏：问题与目标笔记共享的 token 里，最稀有的那个在全语料的 note 级文档频率。
# df <= 这个值，说明问题里抄了一个近乎唯一的指纹词，BM25 可以不检索直接精确匹配。
LEAK_MAX_DF = 2
LEAK_BUCKETS: tuple[tuple[str, int], ...] = (("<=2", 2), ("3-10", 10), ("11-50", 50))

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

# 一个关键词出现在多少篇以上就算"大众"。只用于报告里的参考列，不再作为判据——
# 判据是下面的覆盖率：取"最稀有的那个 token"等于只要问题里有任意一个冷门片段就放行，
# 而中文二字组几乎必然能凑出一个 df=0 的垃圾片段（`注中` 来自"标注中"、`么处` 来自"为什么处理"），
# 所以旧判据 18 条候选 18 条通过，从未拒绝过任何东西。见 docs/reviews/review-001 的 S5。
NEGATIVE_MAX_DOC_FREQ = 5

# 负样本判据：全语料里覆盖该问题内容词最多的那一篇，覆盖率必须低于这个值。
# 定标依据（见 docs/reviews/impl-003）：已知的 18 条负样本覆盖率最高只有 0.545，
# 而 180 条可回答问题的中位数是 0.750；0.60 落在两者之间且高于负样本的观测上界。
# 区分能力 AUC = 0.950；把 180 条可回答问题当作负样本候选喂进来，它能拒掉 75%。
NEGATIVE_MAX_COVERAGE = 0.60


@dataclass
class Stats:
    candidates_read: int = 0
    files_unreadable: int = 0
    kept: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    duplicates: int = 0
    negatives_kept: int = 0
    negatives_dropped: list[tuple[str, str]] = field(default_factory=list)
    leaked: int = 0
    twinned: int = 0


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


def content_tokens(question: str) -> list[str]:
    """问题里的"内容词"：ASCII 词（长度 >= 4）与中文二字组，去重后定序。

    过滤掉 ASCII 短词（the / is / a）与中文单字，它们在任何语料里都到处都是，
    留着只会把覆盖率稀释成噪声。
    """
    return sorted(
        {
            token
            for token in tokenize(question)
            if (token.isascii() and len(token) >= 4) or (not token.isascii() and len(token) == 2)
        }
    )


def note_doc_freq(note_tokens: dict[str, set[str]]) -> Counter[str]:
    """token -> 含有它的**笔记数**（note 级文档频率，不是出现次数）。"""
    df: Counter[str] = Counter()
    for tokens in note_tokens.values():
        df.update(tokens)
    return df


def best_note_coverage(question: str, note_tokens: dict[str, set[str]]) -> tuple[float, str]:
    """返回 (最佳覆盖率, 那篇笔记)：全语料里覆盖该问题内容词最多的一篇覆盖了几成。

    这是负样本的复核判据。它比"最稀有关键词的 df"稳健得多：后者只要问题里有任意一个
    冷门片段就放行；覆盖率问的是"有没有哪一篇真的在谈这个话题"，正是我们想知道的事。
    """
    tokens = content_tokens(question)
    if not tokens:
        return (1.0, "")
    best, best_note = 0.0, ""
    for note, vocabulary in sorted(note_tokens.items()):
        covered = sum(1 for token in tokens if token in vocabulary) / len(tokens)
        if covered > best:
            best, best_note = covered, note
    return (best, best_note)


def leak_min_df(
    question: str, target_tokens: set[str], df: Counter[str], note_count: int
) -> tuple[str, int]:
    """返回 (最稀有的共享 token, 它的 note 级 df)。

    只看**问题与目标笔记都出现**的 token：那才是"照抄"的证据。df 越低，说明抄来的是一个
    近乎唯一的指纹词，BM25 可以不做检索直接精确匹配。

    一个共享 token 都没有时返回 ``("", note_count)`` —— 等价于"和全语料一样常见"，即没有泄漏。
    """
    shared = [token for token in content_tokens(question) if token in target_tokens]
    if not shared:
        return ("", note_count)
    rarest = min(shared, key=lambda token: (df[token], token))
    return (rarest, df[rarest])


def keyword_rarity(question: str, corpus_lower: dict[str, str]) -> tuple[str, int]:
    """返回 (最稀有关键词, 该词出现的文档数)。**仅用于报告里的参考列，不再是判据。**

    ``corpus_lower`` 必须是**小写化**的正文：``tokenize()`` 会 lowercase，拿小写 token 去
    原始大小写的正文里做子串匹配，会把 ``kubernetes`` 记成 df=0，而 ``Kubernetes`` 其实
    出现在 2 篇里。见 docs/reviews/review-001 的 S5。
    """
    tokens = [t for t in tokenize(question) if len(t) >= 2]
    if not tokens:
        return ("", len(corpus_lower))
    rarest, best = "", len(corpus_lower) + 1
    for token in sorted(set(tokens)):
        df = sum(1 for text in corpus_lower.values() if token in text)
        if df < best:
            rarest, best = token, df
    return (rarest, best)


_FLATTENED_NOTE = re.compile(r"^(?P<slug>.*)__[0-9a-f]{8}\.md$")


def slug_index(notes: Iterable[str]) -> dict[str, str]:
    """``slug -> 笔记路径``，用于 O(1) 查找译文孪生。

    语料把源目录压进了文件名并追加一个 8 位内容 hash，所以一对孪生的 hash 不同、slug 相同
    （``foo.zh__aaaaaaaa.md`` 与 ``foo__bbbbbbbb.md``）。不使用这套命名的仓库会得到一个空索引，
    后续的孪生查找就是安全的 no-op。
    """
    index: dict[str, str] = {}
    for note in sorted(notes):
        match = _FLATTENED_NOTE.match(note)
        if match is not None:
            index.setdefault(match.group("slug"), note)
    return index


def twin_of(note: str, index: dict[str, str]) -> str:
    """同一篇笔记的译文孪生，没有就返回空串。"""
    match = _FLATTENED_NOTE.match(note)
    if match is None:
        return ""
    slug = match.group("slug")
    wanted = slug[:-3] if slug.endswith(".zh") else f"{slug}.zh"
    return index.get(wanted, "")


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

    # 语料正文缓存。**必须**用 iter_vault_files 而不是 glob：索引器用的就是它，
    # 两侧规则一旦错位，评测集就会标注到索引里根本没有的笔记，所有模式对那些问题永久记 0。
    # 键也用 `relative_to(vault).as_posix()`，与 ingest 写进 note_path 的字符串逐字一致。
    # 见 docs/reviews/review-001 的 S2 与 S10。
    corpus: dict[str, str] = {}
    for note in iter_vault_files(corpus_dir):
        rel = note.relative_to(corpus_dir).as_posix()
        try:
            corpus[rel] = note.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"  语料读取失败 {rel}: {exc}", file=sys.stderr)
    corpus_normalized = {name: normalize_ws(text) for name, text in corpus.items()}
    corpus_lower = {name: text.lower() for name, text in corpus.items()}
    note_tokens = {name: set(tokenize(text)) for name, text in corpus.items()}
    doc_freq = note_doc_freq(note_tokens)
    twins = slug_index(corpus)

    # 候选文件里的 note_path 是绝对路径，这里按文件名回连到语料的相对路径。
    by_name: dict[str, list[str]] = {}
    for rel in corpus:
        by_name.setdefault(Path(rel).name, []).append(rel)
    print(f"载入语料 {len(corpus)} 篇")

    kept: list[dict[str, object]] = []
    seen: set[str] = set()

    for entry in entries:
        note_path = str(entry.get("note_path", ""))
        note_name = Path(note_path.replace("\\", "/")).name
        question = str(entry.get("question", "")).strip()
        span = str(entry.get("answer_span", "")).strip()
        difficulty = str(entry.get("difficulty", "")).strip().lower()

        matches = by_name.get(note_name, [])
        if len(matches) > 1:
            stats.dropped["源笔记文件名在语料中不唯一"] += 1
            continue
        if not matches:
            stats.dropped["源笔记不在语料中"] += 1
            continue
        note_rel = matches[0]
        if not (MIN_QUESTION_CHARS <= len(question) <= MAX_QUESTION_CHARS):
            stats.dropped["问题长度不合适"] += 1
            continue
        if any(pattern in question for pattern in META_PATTERNS):
            stats.dropped["元问题（无需检索即可回答）"] += 1
            continue
        if not (MIN_SPAN_CHARS <= len(span) <= MAX_SPAN_CHARS):
            stats.dropped["答案片段长度不合适"] += 1
            continue
        if normalize_ws(span) not in corpus_normalized[note_rel]:
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

        leak_token, min_df = leak_min_df(question, note_tokens[note_rel], doc_freq, len(corpus))
        if min_df <= LEAK_MAX_DF:
            stats.leaked += 1
        twin = twin_of(note_rel, twins)
        if twin:
            stats.twinned += 1

        kept.append(
            {
                "question": question,
                "note_path": note_rel,
                "twin_path": twin,
                "answer_span": span,
                "difficulty": difficulty,
                "leak_token": leak_token,
                "leak_min_df": min_df,
            }
        )

    kept.sort(key=lambda item: (str(item["note_path"]), str(item["question"])))
    stats.kept = len(kept)

    # 负样本复核：全语料里覆盖该问题内容词最多的那一篇，覆盖了几成。
    # 覆盖率高 = 某篇笔记其实在谈这个话题 = 这条"不可回答"不可信，丢弃。
    negatives: list[dict[str, object]] = []
    for question in NEGATIVE_QUESTIONS:
        coverage, best_note = best_note_coverage(question, note_tokens)
        token, df = keyword_rarity(question, corpus_lower)
        if coverage < NEGATIVE_MAX_COVERAGE:
            negatives.append(
                {
                    "question": question,
                    "coverage": coverage,
                    "best_note": best_note,
                    "token": token,
                    "df": df,
                }
            )
        else:
            stats.negatives_dropped.append(
                (question, f"`{best_note}` 覆盖了它 {coverage:.0%} 的内容词")
            )
    stats.negatives_kept = len(negatives)

    # 写数据集：正样本用 note 名，负样本 relevant_notes 为空数组
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, item in enumerate(kept, start=1):
        # 分级相关：问题所出自的那一篇 = 2，它的译文孪生 = 1。
        # 孪生同样能回答该问题，把它判为不相关会在换上多语言嵌入模型后制造一次假的 recall 回归。
        grades: dict[str, int] = {str(item["note_path"]): GRADE_SOURCE}
        if item["twin_path"]:
            grades[str(item["twin_path"])] = GRADE_TRANSLATION
        rows.append(
            {
                "id": f"q{index:04d}",
                "question": item["question"],
                "relevant_notes": grades,
                "answerable": True,
                "difficulty": item["difficulty"],
                "answer_span": item["answer_span"],
                "leak": bool(int(str(item["leak_min_df"])) <= LEAK_MAX_DF),
                "leak_token": item["leak_token"],
                "leak_min_df": item["leak_min_df"],
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
                "leak": False,
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
        f"- 有译文孪生（记 grade {GRADE_TRANSLATION}）：**{stats.twinned}** / {stats.kept} 条",
        f"- 标记为词法泄漏（`leak: true`，df <= {LEAK_MAX_DF}）：**{stats.leaked}** / {stats.kept} 条",
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
        f"平均每篇笔记 {stats.kept / max(1, len(per_note)):.2f} 条问题，覆盖 {len(per_note)} 篇笔记"
        f"（全语料 {len(corpus)} 篇，覆盖率 {len(per_note) / max(1, len(corpus)):.1%}）。",
        "",
        "## 词法泄漏分布（问题与目标笔记共享的最稀有 token 的 note 级 df）",
        "",
        "问题是模型读着目标笔记生成的，会把该笔记的稀有标识符逐字抄进去。df 越低，",
        "越是「不检索也能精确匹配」。`leak: true` 的条目**没有被删除**，只是被标记出来，",
        f"这样任何按分组出数的报告都能把它们单列（判据：df <= {LEAK_MAX_DF}）。",
        "",
        "| 共享的最稀有 token 的 df | 条数 |",
        "| --- | --- |",
    ]
    bucket_counts: Counter[str] = Counter()
    for item in kept:
        value = int(str(item["leak_min_df"]))
        label = ">50"
        for name, upper in LEAK_BUCKETS:
            if value <= upper:
                label = name
                break
        bucket_counts[label] += 1
    for name, _ in (*LEAK_BUCKETS, (">50", 0)):
        lines.append(f"| {name} | {bucket_counts.get(name, 0)} |")
    lines.append("")
    if stats.negatives_dropped:
        lines += ["## 被丢弃的负样本（有笔记覆盖了它的内容词，可能其实有答案）", ""]
        for question, reason in stats.negatives_dropped:
            lines.append(f"- {question} —— {reason}")
        lines.append("")
    else:
        lines += [
            "## 被丢弃的负样本",
            "",
            f"无。{len(NEGATIVE_QUESTIONS)} 条候选的最佳笔记覆盖率全部低于 "
            f"{NEGATIVE_MAX_COVERAGE:.0%}（最高 "
            f"{max((float(str(item['coverage'])) for item in negatives), default=0.0):.0%}）。",
            "",
        ]
    lines += [
        "## 负样本清单（已保留）",
        "",
        "`覆盖率` = 全语料里覆盖该问题内容词最多的那一篇覆盖了几成，越低越可信。",
        "`最稀有关键词` 仅供参考，**不再是判据**（它只要问题里有任意一个冷门片段就会放行）。",
        "",
        "| 问题 | 覆盖率 | 覆盖最多的笔记 | 最稀有关键词 | df |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in negatives:
        coverage = float(str(item["coverage"]))
        lines.append(
            f"| {item['question']} | {coverage:.0%} | `{item['best_note']}` "
            f"| `{item['token']}` | {item['df']} |"
        )
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
