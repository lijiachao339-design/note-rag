"""Command line interface: ``note-rag <command>``.

Commands:
    ingest   sync the vault into Postgres (with ``--dry-run`` for a no-DB preview)
    search   run a query against the live index
    eval     run the ablation over a labelled dataset and print a markdown table
    serve    start the HTTP API
    mcp      start the MCP stdio server
    gen-eval generate an eval draft from the corpus (feed it to your agent workflow)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from note_rag import store
from note_rag.bm25 import content_tokens, tokenize
from note_rag.config import get_settings
from note_rag.embed import build_embedder
from note_rag.ingest import chunk_preview, collect_chunks, sync_vault
from note_rag.metrics import (
    abstention_report,
    cluster_bootstrap_ci,
    evaluate,
    ndcg_at_k,
    percentile,
    recall_at_k,
)
from note_rag.retriever import Mode, Retriever, SearchHit

MODES: tuple[Mode, ...] = ("vector", "keyword", "hybrid", "hybrid_rerank")


def _log(message: str) -> None:
    print(message, flush=True)


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.full:
        _log("full re-index requested: the embedding cache will be ignored")

    if args.dry_run:
        from note_rag.ingest import SyncStats

        stats = SyncStats()
        chunks = collect_chunks(
            Path(settings.vault_path),
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars,
            stats=stats,
        )
        _log(f"dry run: {len(chunks)} chunks from {stats.files_scanned} files")
        if stats.failures:
            _log("failures:\n  " + "\n  ".join(stats.failures[:10]))
        if chunks:
            _log(chunk_preview(chunks, limit=args.preview))
        return 0

    conn = store.connect(settings.database_url)
    embedder = build_embedder(
        base_url=settings.embed_base_url,
        api_key=settings.embed_api_key,
        model=settings.embed_model,
        dim=settings.embed_dim,
        batch_size=settings.embed_batch_size,
    )
    try:
        stats = sync_vault(
            settings=settings, conn=conn, embedder=embedder, full=args.full, progress=_log
        )
    finally:
        conn.close()
    if stats.failures:
        _log(f"{len(stats.failures)} file(s) failed to parse (first 3): {stats.failures[:3]}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    settings = get_settings()
    retriever = Retriever.from_settings(settings)
    retriever.build_keyword_index()
    started = time.perf_counter()
    hits = retriever.search(args.query, k=args.k, mode=args.mode)
    elapsed_ms = (time.perf_counter() - started) * 1000
    _log(f"mode={args.mode} k={args.k} took={elapsed_ms:.1f}ms\n")
    for index, hit in enumerate(hits, start=1):
        breadcrumb = " > ".join(hit.heading_path)
        _log(f"{index}. [{hit.score:.4f}] {hit.note_path} :: {breadcrumb or '(root)'}")
        _log(f"    {hit.text[:160].replace(chr(10), ' ')}")
    return 0


def _load_dataset(path: Path) -> list[dict[str, object]]:
    """JSONL rows: ``{"id": "...", "question": "...", "relevant_notes": ["a/b.md", ...]}``."""
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
            if "id" not in row or "question" not in row or "relevant_notes" not in row:
                raise ValueError(f"{path}:{lineno} needs id/question/relevant_notes")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no questions")
    return rows


def _grades_for(row: dict[str, object]) -> dict[str, float]:
    """Relevance grades for one row, as ``{note_path: grade}``.

    ``relevant_notes`` accepts two shapes:

    * a list -- every listed note has grade 1 (and ``[]`` means *unanswerable*: a legal,
      deliberate value used to expose retrievers that always return something);
    * an object ``{note: grade}`` -- graded relevance.

    Grades exist because the corpus is a strict bilingual mirror. The translated twin of a
    note answers the same question, so calling it irrelevant is wrong; but it is not the
    note the question was written from either. Grade 2 = the source note, grade 1 = its
    translation. See docs/reviews/review-001, S8.
    """
    relevant = row["relevant_notes"]
    if isinstance(relevant, dict):
        return {str(note): float(grade) for note, grade in relevant.items()}
    if isinstance(relevant, (list, tuple)):
        return {str(note): 1.0 for note in relevant}
    raise ValueError(f"{row['id']!r}: relevant_notes 必须是数组或 {{笔记: 等级}} 对象")


@dataclass(frozen=True, slots=True)
class ModeReport:
    """One mode's results, with the units kept apart.

    Quality and latency live in separate fields on purpose: flattening them into one dict is
    what turned "every metric is within [0, 1]" into a landmine once p95_ms joined the
    mapping (docs/reviews/review-001, S4).
    """

    quality: dict[str, float]
    abstention: dict[str, float]
    by_lang: dict[str, dict[str, str]]
    latency_ms: dict[str, float]


def _abstain_coverage(question: str, hits: Sequence[SearchHit]) -> float:
    """How much of the question's topical vocabulary the top hit actually covers.

    This is the abstention signal. It deliberately reads the retrieved **text** rather than a
    score, because a usable score is not available on every path: ``vector`` mode reports a
    synthetic rank score and RRF discards score magnitude entirely, so the fused top-1 score
    sits at ~2/61 for every query, answerable or not. Coverage behaves identically for all
    four modes (docs/reviews/review-001, S4).
    """
    tokens = content_tokens(question)
    if not tokens or not hits:
        return 0.0
    vocabulary = set(tokenize(f"{hits[0].title} {hits[0].text}"))
    return sum(1 for token in tokens if token in vocabulary) / len(tokens)


def _ci(values: dict[str, float], cluster_of: dict[str, str], samples: int) -> str:
    """``mean [low, high]`` with the interval bootstrapped over notes, not questions."""
    if not values:
        return "n/a"
    mean = sum(values.values()) / len(values)
    low, high = cluster_bootstrap_ci(values, cluster_of, samples=samples)
    return f"{mean:.3f} [{low:.3f}, {high:.3f}]"


def cmd_eval(args: argparse.Namespace) -> int:
    settings = get_settings()
    dataset = _load_dataset(Path(args.dataset))
    retriever = Retriever.from_settings(settings)
    loaded = retriever.build_keyword_index()
    _log(f"loaded {loaded} chunks; {len(dataset)} questions")

    qrels: dict[str, dict[str, float]] = {str(row["id"]): _grades_for(row) for row in dataset}

    # 标注了却没被索引的笔记 = 该问题对所有模式永久记 0，而且看起来像"检索很差"。
    # 这种错位必须当场炸，不能变成一个漂亮的低分。见 docs/reviews/review-001 的 S10。
    labelled = {note for grades in qrels.values() for note in grades}
    unknown = sorted(labelled - retriever.indexed_notes())
    if unknown:
        raise ValueError(
            f"{len(unknown)} 篇被标注的笔记不在索引里，评测会静默记 0；"
            f"例如 {unknown[:3]}。请先 `note-rag ingest` 或修正 relevant_notes。"
        )

    answerable = {str(row["id"]): bool(qrels[str(row["id"])]) for row in dataset}
    answerable_ids = [qid for qid, flag in answerable.items() if flag]
    unanswerable_ids = [qid for qid, flag in answerable.items() if not flag]
    if not answerable_ids:
        raise ValueError("数据集里没有可回答问题，排序指标无从谈起")
    # 聚簇 bootstrap 的重采样单位：问题所出自的那一篇笔记（grade 最高的那一篇）。
    cluster_of = {qid: max(qrels[qid], key=lambda note: qrels[qid][note]) for qid in answerable_ids}
    lang_of = {str(row["id"]): str(row.get("lang", "unknown")) for row in dataset}
    _log(
        f"  {len(answerable_ids)} 条可回答（来自 {len(set(cluster_of.values()))} 篇笔记）"
        f" + {len(unanswerable_ids)} 条不可回答"
    )

    repeats = max(1, int(args.latency_repeats))
    rankings_by_mode: dict[str, dict[str, list[str]]] = {}
    coverage_by_mode: dict[str, dict[str, float]] = {}
    latency_by_mode: dict[str, list[list[float]]] = {mode: [] for mode in args.modes}
    drifted: list[str] = []

    # 按"重复"而不是按"模式"分层循环：模式在每一轮里交错执行，机器状态的漂移就平摊到所有模式上，
    # 而不是全落在最后一个模式头上。单次串行测量的噪声本来就大于模式间的真实差异。
    for repeat in range(repeats):
        for mode in args.modes:
            latencies: list[float] = []
            rankings: dict[str, list[str]] = {}
            coverage: dict[str, float] = {}
            for row in dataset:
                qid = str(row["id"])
                question = str(row["question"])
                started = time.perf_counter()
                hits = retriever.search_notes(question, k=max(args.ks), mode=mode)
                latencies.append((time.perf_counter() - started) * 1000)
                rankings[qid] = [hit.note_path for hit in hits]
                if repeat == 0:
                    coverage[qid] = _abstain_coverage(question, hits)
            latency_by_mode[mode].append(latencies)
            if repeat == 0:
                rankings_by_mode[mode] = rankings
                coverage_by_mode[mode] = coverage
            elif rankings != rankings_by_mode[mode]:
                drifted.append(f"{mode}@repeat{repeat}")
        _log(f"  repeat {repeat + 1}/{repeats} done")
    if drifted:
        _log(f"WARNING 排名在重复之间发生了变化（检索本应确定性）: {drifted}")

    reports: dict[str, ModeReport] = {}
    for mode in args.modes:
        ranked = rankings_by_mode[mode]
        # 排序质量只统计可回答问题：不可回答问题在 recall/MRR/nDCG 上恒等于 0，
        # 与检索器返回什么完全无关，混进均值只是把所有数字乘上一个常数。见 review-001 的 S4。
        quality = evaluate({qid: ranked[qid] for qid in answerable_ids}, qrels, ks=args.ks)
        abstained = {qid: coverage_by_mode[mode][qid] < args.abstain_threshold for qid in ranked}
        abstention = abstention_report(abstained, answerable)
        # 拒答的代价必须用同一个阈值、同一批查询算出来：被拒答的查询按空排序计分。
        # 只报拒答率，会让"几乎什么都召不回"的检索器看起来最擅长拒答。
        abstention["recall@5_on_answerable"] = sum(
            recall_at_k([] if abstained[qid] else ranked[qid], qrels[qid], 5)
            for qid in answerable_ids
        ) / len(answerable_ids)

        by_lang: dict[str, dict[str, str]] = {}
        for lang in sorted({lang_of[qid] for qid in answerable_ids}):
            ids = [qid for qid in answerable_ids if lang_of[qid] == lang]
            clusters = {qid: cluster_of[qid] for qid in ids}
            by_lang[lang] = {
                "queries": str(len(ids)),
                "notes": str(len(set(clusters.values()))),
                "recall@5": _ci(
                    {qid: recall_at_k(ranked[qid], qrels[qid], 5) for qid in ids},
                    clusters,
                    args.bootstrap_samples,
                ),
                "ndcg@10": _ci(
                    {qid: ndcg_at_k(ranked[qid], qrels[qid], 10) for qid in ids},
                    clusters,
                    args.bootstrap_samples,
                ),
            }

        p50s = [percentile(run, 50) for run in latency_by_mode[mode]]
        p95s = [percentile(run, 95) for run in latency_by_mode[mode]]
        reports[mode] = ModeReport(
            quality=quality,
            abstention=abstention,
            by_lang=by_lang,
            latency_ms={
                "repeats": float(repeats),
                "p50_median": percentile(p50s, 50),
                "p50_min": min(p50s),
                "p50_max": max(p50s),
                "p95_median": percentile(p95s, 50),
                "p95_min": min(p95s),
                "p95_max": max(p95s),
            },
        )

    table = _render_report(args, reports, len(dataset), len(answerable_ids), len(unanswerable_ids))
    _log(table)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table, encoding="utf-8")
        json_path = out_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(
                {
                    "queries": len(dataset),
                    "answerable": len(answerable_ids),
                    "unanswerable": len(unanswerable_ids),
                    "modes": list(args.modes),
                    "latency_repeats": repeats,
                    "abstain_threshold": args.abstain_threshold,
                    "reports": {mode: dataclasses.asdict(r) for mode, r in reports.items()},
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        # 逐条排名也落盘：任何切片（按语言、按难度、按泄漏分组）都能离线复算，
        # 不必为了换一种分组就重跑一次几分钟的评测。
        rankings_path = out_path.with_name(f"{out_path.stem}.rankings.json")
        rankings_path.write_text(
            json.dumps(
                # 逐条覆盖率也存下来：换一个拒答阈值重新出数，不必再跑一遍几分钟的评测。
                {"rankings": rankings_by_mode, "abstain_coverage": coverage_by_mode},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _log(f"wrote {out_path}, {json_path} and {rankings_path}")
    return 0


def _render_report(
    args: argparse.Namespace,
    reports: dict[str, ModeReport],
    total: int,
    answerable: int,
    unanswerable: int,
) -> str:
    """Four tables, because one table mixing units is how a number gets misquoted."""
    modes = list(args.modes)
    header = "| metric | " + " | ".join(modes) + " |"
    divider = "| --- | " + " | ".join("---" for _ in modes) + " |"

    def row(label: str, cells: list[str]) -> str:
        return f"| {label} | " + " | ".join(cells) + " |"

    def quality(metric: str) -> list[str]:
        return [f"{reports[mode].quality[metric]:.4f}" for mode in modes]

    def abstain(key: str) -> list[str]:
        return [f"{reports[mode].abstention.get(key, 0.0):.4f}" for mode in modes]

    def lang_cell(lang: str, metric: str) -> list[str]:
        return [reports[mode].by_lang[lang][metric] for mode in modes]

    def latency(key: str) -> list[str]:
        return [f"{reports[mode].latency_ms[key]:.1f}" for mode in modes]

    def latency_spread(prefix: str) -> list[str]:
        return [
            f"{reports[mode].latency_ms[f'{prefix}_max'] - reports[mode].latency_ms[f'{prefix}_min']:.1f}"
            for mode in modes
        ]

    lines = [
        "# Retrieval ablation",
        "",
        f"Queries: {total} = {answerable} answerable + {unanswerable} unanswerable",
        "",
        f"## 1. 排序质量（只统计 {answerable} 条可回答问题）",
        "",
        header,
        divider,
    ]
    for k in args.ks:
        lines.append(row(f"recall@{k}", quality(f"recall@{k}")))
    lines.append(row("mrr", quality("mrr")))
    for k in args.ks:
        lines.append(row(f"ndcg@{k}", quality(f"ndcg@{k}")))

    lines += [
        "",
        f"## 2. 拒答（top-1 命中的内容词覆盖率 < {args.abstain_threshold:.2f} 即视为拒答）",
        "",
        header,
        divider,
        row(
            f"abstain_rate_on_unanswerable (n={unanswerable})",
            abstain("abstain_rate_on_unanswerable"),
        ),
        row(
            f"false_abstain_rate_on_answerable (n={answerable})",
            abstain("false_abstain_rate_on_answerable"),
        ),
        row(
            f"recall@5_on_answerable @ 同阈值 (n={answerable})",
            abstain("recall@5_on_answerable"),
        ),
        "",
        "三个数必须一起看：对所有问题都拒答，第一行是完美的 1.0，而第三行会掉到 0。",
        "把前两行合成判别力 J = 拒答率 - 误拒率，才是可以跨模式比较的单一数字。",
        "",
        "## 3. 分语言（均值 + 按笔记聚簇的 bootstrap 95% CI）",
        "",
        header,
        divider,
    ]
    langs = sorted({lang for report in reports.values() for lang in report.by_lang})
    for lang in langs:
        lines.append(row(f"{lang} recall@5", lang_cell(lang, "recall@5")))
        lines.append(row(f"{lang} ndcg@10", lang_cell(lang, "ndcg@10")))
        counts = [
            f"{reports[mode].by_lang[lang]['queries']} 题 / {reports[mode].by_lang[lang]['notes']} 篇"
            for mode in modes
        ]
        lines.append(row(f"{lang} 样本量", counts))

    lines += [
        "",
        "重采样单位是**笔记**不是问题：同一篇笔记的 3 条问题高度相关，按问题重采样会低估方差。",
        "样本量列里的「篇」数才是有效样本量。",
        "",
        f"## 4. 延迟（每个模式重复 {int(args.latency_repeats)} 轮，模式在每一轮里交错执行）",
        "",
        header,
        divider,
        row("p50_ms (重复间中位数)", latency("p50_median")),
        row("p50_ms (重复间极差)", latency_spread("p50")),
        row("p95_ms (重复间中位数)", latency("p95_median")),
        row("p95_ms (重复间极差)", latency_spread("p95")),
        "",
        "极差 = 同一模式在不同重复之间的最大差。若两个模式的中位数之差小于各自的极差，",
        "**它们的延迟在本次测量下不可区分**，不要据此下结论。",
        "",
        "## 怎么读这张表",
        "",
        "- `recall@k` 在本数据集上是**真 recall**（多数问题有 2 篇相关笔记：原文 + 译文孪生），",
        "  因此**不能**与 exp-001 的单标签数字直接比较。单标签下它其实是 Success@k / Hit Rate@k。",
        "- `precision@k` 已从报告中移除：它恒等于 `recall@k * |relevant| / k`，不提供独立信息。",
        "- 不可回答问题**不在**第 1 张表里。它们在 recall/MRR/nDCG 上恒为 0，与检索器行为无关，",
        "  混进均值只会把所有数字乘上一个常数。它们的作用在第 2 张表。",
        "",
    ]
    return "\n".join(lines) + "\n"


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("note_rag.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    del args
    from note_rag.mcp_server import main as mcp_main

    mcp_main()
    return 0


def cmd_gen_eval(args: argparse.Namespace) -> int:
    """Emit a chunk sample as JSONL.

    The intended use is delegation: hand each line (or each file) to a cheap model in a
    harness workflow, ask for 3-5 answerable questions plus the source note path, then have
    a strong model spot-check the result. Generated questions are drafts until a human
    confirms the answer really exists in that note.
    """
    settings = get_settings()
    from note_rag.ingest import SyncStats

    stats = SyncStats()
    chunks = collect_chunks(
        Path(settings.vault_path),
        max_chars=settings.chunk_max_chars,
        overlap_chars=settings.chunk_overlap_chars,
        stats=stats,
    )
    by_note: dict[str, list[str]] = {}
    for chunk in chunks:
        by_note.setdefault(chunk.note_path, []).append(chunk.text)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for note_path, texts in by_note.items():
            if args.limit and written >= args.limit:
                break
            payload = {"note_path": note_path, "text": "\n\n".join(texts)[: args.max_chars]}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
    _log(f"wrote {written} note samples to {out_path}")
    _log("next: fan these out to cheap models, then spot-check with a strong model")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="note-rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="sync the vault into Postgres")
    p_ingest.add_argument("--full", action="store_true", help="ignore the incremental cache")
    p_ingest.add_argument("--dry-run", action="store_true", help="parse/chunk only, no database")
    p_ingest.add_argument("--preview", type=int, default=5, help="chunks to preview in dry run")
    p_ingest.set_defaults(func=cmd_ingest)

    p_search = sub.add_parser("search", help="query the index")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)
    p_search.add_argument("--mode", choices=MODES, default="hybrid")
    p_search.set_defaults(func=cmd_search)

    p_eval = sub.add_parser("eval", help="run the retrieval ablation")
    p_eval.add_argument("--dataset", default="eval/dataset.jsonl")
    p_eval.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p_eval.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    p_eval.add_argument("--out", default="eval/out/ablation.md")
    p_eval.add_argument(
        "--latency-repeats",
        type=int,
        default=3,
        help="每个模式重复计时多少轮（默认 3：单轮的噪声大于模式间的真实差异）",
    )
    p_eval.add_argument(
        "--abstain-threshold",
        type=float,
        default=0.30,
        help="top-1 命中的内容词覆盖率低于此值即判为拒答（默认 0.30，见 docs/reviews/impl-004）",
    )
    p_eval.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="按笔记聚簇的 bootstrap 重采样次数（默认 2000）",
    )
    p_eval.set_defaults(func=cmd_eval)

    p_serve = sub.add_parser("serve", help="start the HTTP API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_mcp = sub.add_parser("mcp", help="start the MCP stdio server")
    p_mcp.set_defaults(func=cmd_mcp)

    p_gen = sub.add_parser("gen-eval", help="export note samples for eval generation")
    p_gen.add_argument("--out", default="eval/raw_notes.jsonl")
    p_gen.add_argument("--limit", type=int, default=0, help="0 = all notes")
    p_gen.add_argument("--max-chars", type=int, default=6000)
    p_gen.set_defaults(func=cmd_gen_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        _log("interrupted")
        return 130
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
