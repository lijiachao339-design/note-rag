"""Tests for the eval harness helpers that decide what counts as relevant.

These are pure functions over one dataset row, so they need neither a database nor a corpus.
"""

from __future__ import annotations

import pytest

from note_rag.bm25 import content_tokens
from note_rag.cli import _abstain_coverage, _grades_for, _leak_bucket, _strict_view
from note_rag.retriever import SearchHit


def test_a_list_of_notes_means_every_note_has_grade_one() -> None:
    row = {"id": "q1", "relevant_notes": ["a.md", "b.md"]}

    assert _grades_for(row) == {"a.md": 1.0, "b.md": 1.0}


def test_an_empty_list_is_a_legal_unanswerable_question() -> None:
    """`[]` is deliberate, not missing data: it exposes retrievers that always answer."""
    assert _grades_for({"id": "n1", "relevant_notes": []}) == {}


def test_an_object_carries_graded_relevance() -> None:
    """The source note (2) outranks its translated twin (1); see review-001 S8."""
    row = {"id": "q1", "relevant_notes": {"src.md": 2, "src.zh.md": 1}}

    assert _grades_for(row) == {"src.md": 2.0, "src.zh.md": 1.0}


def test_a_scalar_is_rejected_rather_than_silently_scored_zero() -> None:
    with pytest.raises(ValueError, match="relevant_notes"):
        _grades_for({"id": "q1", "relevant_notes": "a.md"})


def _hit(title: str, text: str) -> SearchHit:
    return SearchHit(
        chunk_id="c1", note_path="n.md", title=title, heading_path=(), text=text, score=1.0
    )


def test_content_tokens_drops_the_words_that_are_everywhere() -> None:
    """Short ASCII words and CJK unigrams occur in nearly every note, so they are noise."""
    tokens = content_tokens("How do the retriever and BM25 handle 混合检索?")

    assert "retriever" in tokens
    assert "混合" in tokens
    assert "the" not in tokens, "3 字母的功能词不该算内容词"
    assert "混" not in tokens, "中文单字不该算内容词"


# The value the `note-rag eval --abstain-threshold` default ships with; see docs/reviews/impl-004.
ABSTAIN_THRESHOLD = 0.30


def test_abstain_coverage_separates_an_on_topic_hit_from_an_off_topic_one() -> None:
    """Both sides of the shipped threshold, on the same retrieved passage.

    An on-topic question does not score near 1.0 even when the passage answers it: the CJK
    tokenizer emits sliding bigrams, and the ones straddling a word boundary ("么融" out of
    "怎么融合") are absent from any note. That is exactly why the default threshold is 0.30
    rather than something that looks tidier.
    """
    hits = [_hit("混合检索", "向量检索与 BM25 用 RRF 融合两路排名。")]

    on_topic = _abstain_coverage("BM25 和 RRF 怎么融合排名？", hits)
    off_topic = _abstain_coverage("InnoDB 的 next-key lock 如何避免幻读？", hits)

    assert on_topic >= ABSTAIN_THRESHOLD, "命中真的覆盖了问题，不该被判为拒答"
    assert off_topic < ABSTAIN_THRESHOLD, "跑题的命中必须被判为拒答"
    assert on_topic > off_topic


def test_abstain_coverage_of_no_hits_is_zero() -> None:
    assert _abstain_coverage("任何问题", []) == 0.0


def test_strict_view_keeps_only_the_note_the_question_came_from() -> None:
    """The twin is genuinely relevant, but counting it caps recall at ~0.5 for a retriever
    that cannot cross languages -- a property of the corpus, not of the ranking."""
    qrels = {"q1": {"src.md": 2.0, "src.zh.md": 1.0}, "q2": {"solo.md": 2.0}}

    assert _strict_view(qrels) == {"q1": {"src.md": 2.0}, "q2": {"solo.md": 2.0}}


def test_strict_view_leaves_unanswerable_queries_empty() -> None:
    """They must stay excluded from the ranking tables under either view."""
    assert _strict_view({"n1": {}}) == {"n1": {}}


def test_strict_view_keeps_every_document_tied_at_the_top_grade() -> None:
    """Two equally-primary notes is a labelling shape we should not silently drop half of."""
    qrels = {"q1": {"a.md": 2.0, "b.md": 2.0, "c.md": 1.0}}

    assert _strict_view(qrels) == {"q1": {"a.md": 2.0, "b.md": 2.0}}


def test_leak_buckets_match_the_boundaries_the_dataset_report_uses() -> None:
    """The eval table and eval/dataset.report.md must bucket identically or they disagree."""
    assert [_leak_bucket(df) for df in (0, 2, 3, 10, 11, 50, 51, 9999)] == [
        "<=2",
        "<=2",
        "3-10",
        "3-10",
        "11-50",
        "11-50",
        ">50",
        ">50",
    ]
