from __future__ import annotations

import math

import pytest

from note_rag.metrics import (
    evaluate,
    format_report,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_perfect_and_partial() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, 2) == 1.0
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, 3) == 0.5
    assert recall_at_k(["x", "y", "a"], {"a"}, 2) == 0.0


def test_recall_with_no_relevant_documents_is_zero_not_error() -> None:
    # Unanswerable questions must stay in the average, otherwise a retriever that
    # refuses to answer looks artificially perfect.
    assert recall_at_k(["a", "b"], set(), 5) == 0.0
    assert reciprocal_rank(["a"], set()) == 0.0


def test_precision_uses_retrieved_length() -> None:
    assert precision_at_k(["a", "x"], {"a"}, 2) == 0.5
    assert precision_at_k([], {"a"}, 5) == 0.0


def test_reciprocal_rank_uses_first_hit() -> None:
    assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert reciprocal_rank(["a"], {"a"}) == 1.0


def test_ndcg_is_one_for_a_perfect_ranking() -> None:
    assert ndcg_at_k(["a", "b"], {"a": 2.0, "b": 1.0}, 5) == pytest.approx(1.0)


def test_ndcg_rewards_putting_the_most_relevant_first() -> None:
    good = ndcg_at_k(["a", "b"], {"a": 2.0, "b": 1.0}, 5)
    bad = ndcg_at_k(["b", "a"], {"a": 2.0, "b": 1.0}, 5)
    assert good > bad
    assert 0.0 <= bad < 1.0


def test_ndcg_is_zero_when_nothing_relevant_is_retrieved() -> None:
    assert ndcg_at_k(["x", "y"], {"a": 1.0}, 5) == 0.0


def test_evaluate_returns_flat_comparable_keys() -> None:
    rankings = {"q1": ["a", "b"], "q2": ["x", "y"]}
    qrels = {"q1": {"a"}, "q2": {"y"}}
    report = evaluate(rankings, qrels, ks=(1, 2))
    assert report["queries"] == 2
    assert report["recall@1"] == pytest.approx(0.5)
    assert report["recall@2"] == pytest.approx(1.0)
    assert report["mrr"] == pytest.approx((1.0 + 0.5) / 2)
    assert set(report) >= {"recall@1", "precision@1", "ndcg@1", "mrr"}


def test_evaluate_rejects_missing_qrels_and_empty_rankings() -> None:
    with pytest.raises(ValueError, match="qrels missing"):
        evaluate({"q1": ["a"]}, {})
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate({}, {})


def test_evaluate_accepts_graded_relevance() -> None:
    report = evaluate({"q1": ["a"]}, {"q1": {"a": 2.0}})
    assert report["ndcg@5"] == pytest.approx(1.0)


def test_percentile_uses_nearest_rank() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(values, 50) == 30.0
    assert percentile(values, 95) == 50.0
    assert percentile(values, 0) == 10.0
    with pytest.raises(ValueError, match="must not be empty"):
        percentile([], 50)
    with pytest.raises(ValueError, match="p must be within"):
        percentile(values, 101)


def test_format_report_is_valid_markdown() -> None:
    text = format_report({"queries": 12.0, "recall@5": 0.8734}, title="hybrid")
    assert "### hybrid" in text
    assert "| queries | 12 |" in text
    assert "| recall@5 | 0.8734 |" in text


def test_metrics_are_bounded() -> None:
    rankings = {"q": ["a", "b", "c"]}
    report = evaluate(rankings, {"q": {"a", "c"}}, ks=(1, 2, 3))
    for key, value in report.items():
        if key == "queries":
            continue
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0
