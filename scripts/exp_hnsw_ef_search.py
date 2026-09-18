#!/usr/bin/env python
"""实验 EXP-002：HNSW 的 `ef_search` 是否在压低 `vector` 一列？

背景（来自 Claude Code + Opus 的审查 S3）：

    `store.py` 用 HNSW 建索引（近似检索），`retriever.py` 请求 `limit=candidate_k=50`，
    而 pgvector 的 `hnsw.ef_search` 默认只有 **40**。
    ef_search 小于请求的 limit 时召回会明显掉。
    Opus 用**精确余弦**离线复现 `vector` 一列，得到 recall@10 = 0.4091，
    而记录的（走 Postgres）是 0.3182 —— 差 9.1 个点。
    它明确说"这一条请先验证再下结论（我没有数据库连接）"。

本脚本就是那个验证：固定测试集与嵌入，只改 `hnsw.ef_search`，看 `vector` 一列怎么动。

用法：
    uv run python scripts/exp_hnsw_ef_search.py
    uv run python scripts/exp_hnsw_ef_search.py --vals 1 40 100 200 400 1000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from note_rag import store
from note_rag.config import get_settings
from note_rag.embed import build_embedder
from note_rag.metrics import evaluate, percentile

DATASET = Path("eval/dataset.jsonl")

# 记录在 eval/out/ablation.md 里的 vector 一列（走 Postgres，ef_search 默认 40）
RECORDED = {"recall@1": 0.1768, "recall@5": 0.2677, "recall@10": 0.3182, "mrr": 0.2216}
# Opus 用精确余弦离线复现的 vector 一列（上界参考）
EXACT_UPPER_BOUND = {"recall@1": 0.1919, "recall@5": 0.3232, "recall@10": 0.4091, "mrr": 0.2570}


def load_rows() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def vector_rankings(conn, embedder, rows, *, top_k: int) -> dict[str, list[str]]:
    """按 note 粒度复现 `Retriever.search_notes(mode='vector')` 的排序逻辑。

    有意复用生产代码的**同一套**去重规则（每篇取最佳 chunk），
    否则测出来的差异就无法归因到 ef_search。
    """
    rankings: dict[str, list[str]] = {}
    for row in rows:
        [vector] = embedder.embed([str(row["question"])])
        hits = store.vector_search(conn, vector, limit=top_k)
        # 查 chunk -> note 的映射
        mapping = store.note_paths_for(conn, [chunk_id for chunk_id, _ in hits])
        seen: list[str] = []
        for chunk_id, _score in hits:
            note = mapping.get(chunk_id)
            if note and note not in seen:
                seen.append(note)
        rankings[str(row["id"])] = seen
    return rankings


def _strict_grades(relevant: object) -> dict[str, float]:
    """``relevant_notes`` -> ``{note: grade}``，只保留 grade 最高的那些（strict 视图）。

    接受两种形状：数组（旧格式，全部 grade 1）与 ``{笔记: 等级}`` 字典（当前格式）。
    与 `note_rag.cli._strict_view` 是同一个口径 —— 两边不一致的话，这个脚本报出来的
    recall 就没法和 `eval/out/ablation.md` 放在一起看。
    """
    if isinstance(relevant, dict):
        grades = {str(note): float(grade) for note, grade in relevant.items()}
    elif isinstance(relevant, (list, tuple)):
        grades = {str(note): 1.0 for note in relevant}
    else:
        raise ValueError(f"relevant_notes 形状不对: {type(relevant).__name__}")
    if not grades:
        return {}
    top = max(grades.values())
    return {note: grade for note, grade in grades.items() if grade == top}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="测试 hnsw.ef_search 对 vector 召回的影响")
    parser.add_argument("--vals", type=int, nargs="+", default=[40, 100, 200, 400])
    parser.add_argument("--top-k", type=int, default=50, help="与 candidate_k 一致")
    args = parser.parse_args(argv)

    settings = get_settings()
    rows = load_rows()
    answerable = [row for row in rows if row.get("answerable")]
    # 口径必须与 `note-rag eval` 的 **strict 视图**一致：只认问题所出自的那一篇（grade 最高）。
    # 这里原来写的是 `list(row["relevant_notes"])` —— 对一个 {笔记: 等级} 字典取 list 只会拿到键，
    # grade 被丢掉，于是所有笔记都成了 grade 1，测的是另一套口径，数字与评测不可比。
    qrels = {str(row["id"]): _strict_grades(row["relevant_notes"]) for row in rows}
    print(
        f"数据集 {len(rows)} 行（可回答 {len(answerable)}）；口径：strict（只认原文，与 eval 一致）"
    )

    conn = store.connect(settings.database_url)
    embedder = build_embedder(
        base_url=settings.embed_base_url,
        api_key=settings.embed_api_key,
        model=settings.embed_model,
        dim=settings.embed_dim,
        batch_size=settings.embed_batch_size,
    )
    print(f"语料：{store.corpus_stats(conn)}  嵌入维度 {embedder.dim}")

    results: dict[int, dict[str, float]] = {}
    try:
        for ef in args.vals:
            with conn.cursor() as cur:
                cur.execute(
                    f"SET hnsw.ef_search = {int(ef)}"
                )  # 只能是字面量，故用 f-string（值已转 int）
            started = time.perf_counter()
            rankings = vector_rankings(conn, embedder, rows, top_k=args.top_k)
            elapsed = time.perf_counter() - started
            report = evaluate(rankings, qrels, ks=(1, 5, 10))
            report["wall_s"] = elapsed
            results[ef] = report
            print(
                f"  ef_search={ef:<5} R@1={report['recall@1']:.4f} R@5={report['recall@5']:.4f} "
                f"R@10={report['recall@10']:.4f} MRR={report['mrr']:.4f}  ({elapsed:.1f}s)"
            )

        # 再单独测一次可回答子集 + 延迟，供对照
        with conn.cursor() as cur:
            cur.execute("SET hnsw.ef_search = 200")
        latencies: list[float] = []
        subset_rankings: dict[str, list[str]] = {}
        for row in answerable:
            [vector] = embedder.embed([str(row["question"])])
            t0 = time.perf_counter()
            hits = store.vector_search(conn, vector, limit=args.top_k)
            latencies.append((time.perf_counter() - t0) * 1000)
            mapping = store.note_paths_for(conn, [chunk_id for chunk_id, _ in hits])
            seen: list[str] = []
            for chunk_id, _score in hits:
                note = mapping.get(chunk_id)
                if note and note not in seen:
                    seen.append(note)
            subset_rankings[str(row["id"])] = seen
        subset = evaluate(subset_rankings, {k: qrels[k] for k in subset_rankings}, ks=(1, 5, 10))
        print(
            f"  ef_search=200 可回答子集: R@1={subset['recall@1']:.4f} R@5={subset['recall@5']:.4f} "
            f"R@10={subset['recall@10']:.4f} MRR={subset['mrr']:.4f} "
            f"p95={percentile(latencies, 95):.1f}ms"
        )
    finally:
        conn.close()

    print()
    print("=== 对照表 ===")
    print("| ef_search | recall@1 | recall@5 | recall@10 | mrr |")
    print("| --- | --- | --- | --- | --- |")
    print(
        f"| 记录值(默认40) | {RECORDED['recall@1']:.4f} | {RECORDED['recall@5']:.4f} | "
        f"{RECORDED['recall@10']:.4f} | {RECORDED['mrr']:.4f} |"
    )
    for ef, report in results.items():
        print(
            f"| {ef} | {report['recall@1']:.4f} | {report['recall@5']:.4f} | "
            f"{report['recall@10']:.4f} | {report['mrr']:.4f} |"
        )
    print(
        f"| 精确余弦(离线上界) | {EXACT_UPPER_BOUND['recall@1']:.4f} | {EXACT_UPPER_BOUND['recall@5']:.4f} | "
        f"{EXACT_UPPER_BOUND['recall@10']:.4f} | {EXACT_UPPER_BOUND['mrr']:.4f} |"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
