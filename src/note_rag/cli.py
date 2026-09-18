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
import json
import sys
import time
from pathlib import Path

from note_rag import store
from note_rag.config import get_settings
from note_rag.embed import build_embedder
from note_rag.ingest import chunk_preview, collect_chunks, sync_vault
from note_rag.metrics import evaluate, format_report, percentile
from note_rag.retriever import Mode, Retriever

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


def cmd_eval(args: argparse.Namespace) -> int:
    settings = get_settings()
    dataset = _load_dataset(Path(args.dataset))
    retriever = Retriever.from_settings(settings)
    loaded = retriever.build_keyword_index()
    _log(f"loaded {loaded} chunks; {len(dataset)} questions\n")

    qrels: dict[str, dict[str, float]] = {str(row["id"]): _grades_for(row) for row in dataset}

    # 标注了却没被索引的笔记 = 该问题对所有模式永久记 0，而且看起来像“检索很差”。
    # 这种错位必须当场炸，不能变成一个漂亮的低分。见 docs/reviews/review-001 的 S10。
    labelled = {note for grades in qrels.values() for note in grades}
    unknown = sorted(labelled - retriever.indexed_notes())
    if unknown:
        raise ValueError(
            f"{len(unknown)} 篇被标注的笔记不在索引里，评测会静默记 0；"
            f"例如 {unknown[:3]}。请先 `note-rag ingest` 或修正 relevant_notes。"
        )

    reports: dict[str, dict[str, float]] = {}
    latencies: dict[str, list[float]] = {}
    all_rankings: dict[str, dict[str, list[str]]] = {}

    for mode in args.modes:
        rankings: dict[str, list[str]] = {}
        latencies[mode] = []
        for row in dataset:
            qid = str(row["id"])
            question = str(row["question"])
            started = time.perf_counter()
            hits = retriever.search_notes(question, k=max(args.ks), mode=mode)
            latencies[mode].append((time.perf_counter() - started) * 1000)
            rankings[qid] = [hit.note_path for hit in hits]
        report = evaluate(rankings, qrels, ks=args.ks)
        report["p50_ms"] = percentile(latencies[mode], 50)
        report["p95_ms"] = percentile(latencies[mode], 95)
        reports[mode] = report
        all_rankings[mode] = rankings

    lines = ["# Retrieval ablation", "", f"Queries: {len(dataset)}", ""]
    metrics = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10", "p50_ms", "p95_ms"]
    lines.append("| metric | " + " | ".join(args.modes) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in args.modes) + " |")
    for metric in metrics:
        if all(metric in reports[mode] for mode in args.modes):
            cells = " | ".join(f"{reports[mode][metric]:.4f}" for mode in args.modes)
            lines.append(f"| {metric} | {cells} |")

    table = "\n".join(lines) + "\n"
    _log(table)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table, encoding="utf-8")
        # 原始指标同时落盘为 JSON：只写 markdown 表会让实验文档里的某些行无法从提交物复现
        # （p50_ms 等只出现在 stdout 的报告里）。见 docs/reviews/review-001 的 S9。
        json_path = out_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(
                {"queries": len(dataset), "modes": list(args.modes), "reports": reports},
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
            json.dumps(all_rankings, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _log(f"wrote {out_path}, {json_path} and {rankings_path}")
        for mode, report in reports.items():
            _log(format_report(report, title=mode))
    return 0


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
