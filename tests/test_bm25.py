from __future__ import annotations

import pytest

from note_rag.bm25 import BM25, tokenize


def test_tokenize_keeps_ascii_words_and_splits_cjk() -> None:
    tokens = tokenize("Python asyncio 事件循环")
    assert "python" in tokens
    assert "asyncio" in tokens
    # Chinese is indexed as unigrams plus bigrams (no segmentation dictionary needed).
    assert "事" in tokens
    assert "事件" in tokens
    assert "事件循" in tokens or "件循" in tokens


def test_tokenize_normalises_case_and_punctuation() -> None:
    assert tokenize("FastAPI, FastAPI!") == ["fastapi", "fastapi"]


def test_exact_match_ranks_first() -> None:
    index = BM25()
    index.add("d1", "postgres pgvector 向量检索")
    index.add("d2", "redis 缓存与限流")
    index.add("d3", "无关的一篇笔记")
    results = index.search("pgvector 向量")
    assert results
    assert results[0][0] == "d1"


def test_rare_terms_have_higher_idf() -> None:
    index = BM25()
    for i in range(50):
        index.add(f"common{i}", "笔记 记录 日常")
    index.add("rare", "笔记 记录 hnsw 索引")
    assert index.idf("hnsw") > index.idf("笔记")


def test_empty_query_returns_nothing() -> None:
    index = BM25()
    index.add("d1", "hello world")
    assert index.search("") == []
    assert index.search("完全不存在的词xyzzy") == []


def test_remove_updates_the_index() -> None:
    index = BM25()
    index.add("d1", "alpha beta")
    index.add("d2", "gamma delta")
    assert len(index) == 2
    index.remove("d1")
    assert len(index) == 1
    assert index.search("alpha") == []
    index.remove("d1")  # idempotent
    assert len(index) == 1


def test_reindex_same_id_replaces_the_old_text() -> None:
    index = BM25()
    index.add("d1", "old content about redis")
    index.add("d1", "new content about postgres")
    assert len(index) == 1
    assert index.search("redis") == []
    assert index.search("postgres")[0][0] == "d1"


def test_length_normalisation_prefers_the_focused_document() -> None:
    index = BM25(b=0.75)
    index.add("focused", "pgvector")
    index.add("padded", "pgvector " + " ".join(["filler"] * 400))
    assert index.search("pgvector")[0][0] == "focused"


def test_limit_and_ordering() -> None:
    index = BM25()
    for i in range(10):
        index.add(f"d{i}", f"term common{i}" if i else "term term term")
    results = index.search("term", limit=3)
    assert len(results) == 3
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="k1 must be positive"):
        BM25(k1=0)
    with pytest.raises(ValueError, match="b must be within"):
        BM25(b=1.5)


def test_top_terms_orders_by_idf() -> None:
    index = BM25()
    for i in range(20):
        index.add(f"d{i}", "公开 内容")
    index.add("rare", "公开 hnsw")
    terms = index.top_terms("公开 hnsw")
    assert terms[0] == "hnsw"
