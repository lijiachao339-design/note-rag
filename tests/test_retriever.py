"""Retriever tests. No database: the store layer is monkeypatched with fixed rows.

The first test in this file is a regression test for a bug that shipped and was only caught
by exercising the HTTP API by hand: ``AppState`` created the retriever but never built the
keyword index, and ``_hydrate`` degrades silently for unknown chunk ids -- so every hit came
back with an empty title/text/note_path while health checks and metrics stayed green.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from note_rag import store
from note_rag.embed import HashingEmbedder
from note_rag.retriever import Reranker, Retriever, SearchHit


class FakeConn:
    def close(self) -> None:
        pass

    def rollback(self) -> None:
        pass


def _rows() -> list[store.ChunkRow]:
    return [
        store.ChunkRow(
            id="c1",
            note_path="20-知识/检索/混合检索.md",
            title="混合检索",
            heading_path=("混合检索",),
            text="向量检索与 BM25 各有所长，用 RRF 融合两路排名。",
        ),
        store.ChunkRow(
            id="c2",
            note_path="20-知识/检索/混合检索.md",
            title="混合检索",
            heading_path=("混合检索", "评测"),
            text="recall@5 与 MRR 是最常用的检索评测指标。",
        ),
        store.ChunkRow(
            id="c3",
            note_path="20-知识/检索/BM25.md",
            title="BM25",
            heading_path=("BM25",),
            text="BM25 的长度归一化参数 b 控制长文档的惩罚强度。",
        ),
    ]


@pytest.fixture
def retriever(monkeypatch: pytest.MonkeyPatch) -> Retriever:
    monkeypatch.setattr(store, "load_chunk_rows", lambda conn: _rows())
    monkeypatch.setattr(store, "corpus_stats", lambda conn: {"chunks": 3, "notes": 2})
    return Retriever(conn=FakeConn(), embedder=HashingEmbedder(dim=32))


def test_search_hydrates_hits_without_an_explicit_index_build(retriever: Retriever) -> None:
    """Regression: search() must be self-sufficient. The API server forgot to build the
    index and returned structurally empty hits."""
    hits = retriever.search("RRF 融合", k=2, mode="keyword")

    assert hits, "关键词检索必须能命中"
    assert hits[0].note_path == "20-知识/检索/混合检索.md"
    assert hits[0].text, "命中必须带上正文，而不是空字符串"
    assert hits[0].heading_path == ("混合检索",)


def test_index_is_built_lazily_only_once(
    retriever: Retriever, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}
    original = store.load_chunk_rows

    def counting(conn: Any) -> list[store.ChunkRow]:
        calls["n"] += 1
        return original(conn)

    monkeypatch.setattr(store, "load_chunk_rows", counting)

    retriever.search("RRF", k=1, mode="keyword")
    retriever.search("BM25", k=1, mode="keyword")

    assert calls["n"] == 1, "索引只应构建一次，不能每次检索都全量加载"


def test_vendor_mode_ranks_by_the_store_order(
    retriever: Retriever, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        store, "vector_search", lambda conn, vec, limit: [("c3", 0.91), ("c1", 0.42)]
    )

    hits = retriever.search("任意查询", k=2, mode="vector")

    assert [hit.chunk_id for hit in hits] == ["c3", "c1"]
    assert all(hit.text for hit in hits), "向量检索的命中同样必须被 hydrate"


def test_hybrid_is_deterministic_and_hydrated(
    retriever: Retriever, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, "vector_search", lambda conn, vec, limit: [("c2", 0.9), ("c3", 0.8)])

    first = retriever.search("RRF 评测指标", k=3, mode="hybrid")
    second = retriever.search("RRF 评测指标", k=3, mode="hybrid")

    assert [hit.chunk_id for hit in first] == [hit.chunk_id for hit in second]
    assert all(hit.text and hit.note_path for hit in first)


class ReverseReranker(Reranker):
    name = "reverse"

    def rerank(self, query: str, candidates: Sequence[SearchHit]) -> list[SearchHit]:
        return list(reversed(candidates))


def test_hybrid_rerank_applies_the_reranker(
    retriever: Retriever, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, "vector_search", lambda conn, vec, limit: [("c1", 0.9)])

    plain = retriever.search("RRF", k=3, mode="hybrid")
    retriever.reranker = ReverseReranker()
    reranked = retriever.search("RRF", k=3, mode="hybrid_rerank")

    assert [hit.chunk_id for hit in reranked] == [hit.chunk_id for hit in plain][::-1]


def test_search_notes_keeps_only_the_best_chunk_per_note(retriever: Retriever) -> None:
    hits = retriever.search_notes("检索 评测 recall", k=5, mode="keyword")

    paths = [hit.note_path for hit in hits]
    assert paths, "应当至少命中一篇笔记"
    assert len(paths) == len(set(paths)), "note 粒度结果不得出现重复笔记"
    assert "20-知识/检索/混合检索.md" in paths


def test_corpus_and_index_stats_are_reported(retriever: Retriever) -> None:
    retriever.build_keyword_index()

    assert retriever.index_stats() == {"keyword_index_docs": 3, "chunks_cached": 3}
    assert retriever.corpus_stats() == {"chunks": 3, "notes": 2}


def test_invalid_arguments_are_rejected(retriever: Retriever) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        retriever.search("   ", k=1, mode="keyword")
    with pytest.raises(ValueError, match="k must be positive"):
        retriever.search("RRF", k=0, mode="keyword")
