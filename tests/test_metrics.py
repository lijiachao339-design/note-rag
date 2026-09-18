from __future__ import annotations

import math

import pytest

from note_rag.metrics import (
    abstention_report,
    cluster_bootstrap_ci,
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
    assert set(report) >= {"recall@1", "ndcg@1", "mrr"}


def test_evaluate_omits_precision_because_it_carries_no_extra_information() -> None:
    """precision@k is recall@k * |relevant| / k -- a constant multiple, so it is not reported.

    The function stays available for callers who want it; what it must not do is occupy a
    column in the ablation table pretending to be independent evidence (review-001, S6).
    """
    rankings = {"q1": ["a", "x", "b"]}
    qrels: dict[str, dict[str, float]] = {"q1": {"a": 2.0, "b": 1.0}}
    report = evaluate(rankings, qrels, ks=(3,))

    assert "precision@3" not in report
    relevant = len(qrels["q1"])
    assert precision_at_k(rankings["q1"], qrels["q1"], 3) == pytest.approx(
        report["recall@3"] * relevant / 3
    )


def test_unanswerable_questions_score_zero_for_every_possible_ranking() -> None:
    """They are a constant, not a measurement -- which is why they are reported separately.

    An earlier comment here claimed keeping them in the average is how you catch a retriever
    that refuses to answer. It is not: returning nothing and inventing ten hits both score
    0.0, so the average only shifts by a fixed factor. See abstention_report and
    docs/reviews/review-001, S4.
    """
    assert recall_at_k([], set(), 5) == 0.0
    assert recall_at_k(["a", "b", "c"], set(), 5) == 0.0
    assert reciprocal_rank([], set()) == 0.0
    assert reciprocal_rank(["a"], set()) == 0.0
    assert ndcg_at_k(["a", "b"], set(), 5) == 0.0


def test_abstention_report_pairs_the_win_with_its_cost() -> None:
    abstained = {"q1": False, "q2": True, "n1": True, "n2": False}
    answerable = {"q1": True, "q2": True, "n1": False, "n2": False}

    report = abstention_report(abstained, answerable)

    assert report["unanswerable"] == 2
    assert report["answerable"] == 2
    assert report["abstain_rate_on_unanswerable"] == pytest.approx(0.5)
    assert report["false_abstain_rate_on_answerable"] == pytest.approx(0.5)


def test_abstaining_on_everything_looks_perfect_on_one_number_and_terrible_on_the_other() -> None:
    """The reason the two numbers must always be reported together."""
    abstained = {"q1": True, "n1": True}
    report = abstention_report(abstained, {"q1": True, "n1": False})

    assert report["abstain_rate_on_unanswerable"] == 1.0
    assert report["false_abstain_rate_on_answerable"] == 1.0


def test_abstention_report_rejects_queries_with_no_answerable_flag() -> None:
    with pytest.raises(ValueError, match="answerable flag missing"):
        abstention_report({"q1": True}, {})


def test_cluster_bootstrap_ci_brackets_the_mean_and_is_reproducible() -> None:
    per_query = {f"q{i}": float(i % 2) for i in range(40)}
    clusters = {f"q{i}": f"note{i // 4}" for i in range(40)}

    low, high = cluster_bootstrap_ci(per_query, clusters, samples=500)

    assert low <= 0.5 <= high
    assert (low, high) == cluster_bootstrap_ci(per_query, clusters, samples=500)


def test_cluster_bootstrap_ci_is_wider_than_ignoring_the_clustering() -> None:
    """Questions from one note share their fate, so treating them as independent lies.

    Here every note is internally unanimous, which is the worst case: the per-question
    interval collapses while the clustered one keeps the real uncertainty.
    """
    per_query = {f"q{i}": float((i // 3) % 2) for i in range(60)}
    clustered = {f"q{i}": f"note{i // 3}" for i in range(60)}
    as_if_independent = {f"q{i}": f"q{i}" for i in range(60)}

    low_c, high_c = cluster_bootstrap_ci(per_query, clustered, samples=1000)
    low_i, high_i = cluster_bootstrap_ci(per_query, as_if_independent, samples=1000)

    assert (high_c - low_c) > (high_i - low_i)


def test_cluster_bootstrap_ci_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        cluster_bootstrap_ci({}, {})
    with pytest.raises(ValueError, match="samples must be positive"):
        cluster_bootstrap_ci({"q": 1.0}, {"q": "n"}, samples=0)
    with pytest.raises(ValueError, match="confidence must be within"):
        cluster_bootstrap_ci({"q": 1.0}, {"q": "n"}, confidence=1.0)
    with pytest.raises(ValueError, match="cluster missing"):
        cluster_bootstrap_ci({"q": 1.0}, {})


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
