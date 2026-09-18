from __future__ import annotations

import pytest

from note_rag.fusion import DEFAULT_RRF_K, interleave, reciprocal_rank_fusion, rrf_scores


def test_document_in_both_lists_outranks_single_list_document() -> None:
    fused = dict(reciprocal_rank_fusion({"vector": ["a", "b", "c"], "keyword": ["b", "d", "a"]}))
    assert fused["a"] > fused["c"]
    assert fused["b"] > fused["d"]
    # "a" is 1st + 3rd, "b" is 2nd + 1st: b must win.
    assert fused["b"] > fused["a"]


def test_scores_follow_the_rrf_formula() -> None:
    fused = rrf_scores({"vector": ["x", "y"]}, k=60)
    assert fused["x"] == pytest.approx(1 / 61)
    assert fused["y"] == pytest.approx(1 / 62)


def test_smaller_k_sharpens_the_top_of_the_list() -> None:
    rankings = {"a": ["first", "second"]}
    sharp = rrf_scores(rankings, k=1)
    flat = rrf_scores(rankings, k=1000)
    assert sharp["first"] / sharp["second"] > flat["first"] / flat["second"]


def test_ordering_is_deterministic_for_ties() -> None:
    rankings = {"vector": ["b", "a"], "keyword": ["a", "b"]}
    first = reciprocal_rank_fusion(rankings)
    second = reciprocal_rank_fusion(rankings)
    assert first == second
    # Identical scores -> tie broken by best rank, then id.
    assert [doc for doc, _ in first] == ["a", "b"]


def test_zero_weight_drops_a_retriever() -> None:
    rankings = {"vector": ["v"], "keyword": ["k"]}
    fused = rrf_scores(rankings, weights={"keyword": 0.0})
    assert "k" not in fused
    assert "v" in fused


def test_weights_change_which_retriever_dominates() -> None:
    # "v_first" is 1st for the vector retriever but last for keyword, and vice versa.
    # Reweighting the fusion must flip which one wins.
    rankings = {
        "vector": ["v_first", "x", "y", "z", "k_first"],
        "keyword": ["k_first", "x", "y", "z", "v_first"],
    }
    heavy_vector = rrf_scores(rankings, weights={"vector": 10.0, "keyword": 0.1})
    heavy_keyword = rrf_scores(rankings, weights={"vector": 0.1, "keyword": 10.0})
    assert heavy_vector["v_first"] > heavy_vector["k_first"]
    assert heavy_keyword["k_first"] > heavy_keyword["v_first"]


def test_top_ranked_everywhere_is_weight_invariant() -> None:
    # Sanity check on the RRF formula: a doc that is rank 1 in every list has the same score
    # regardless of the relative weights -- only its margin over the others changes.
    rankings = {"vector": ["shared", "v"], "keyword": ["shared", "k"]}
    a = rrf_scores(rankings, weights={"vector": 0.1, "keyword": 10.0})
    b = rrf_scores(rankings, weights={"vector": 10.0, "keyword": 0.1})
    assert a["shared"] == pytest.approx(b["shared"])


def test_negative_weight_and_bad_k_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        reciprocal_rank_fusion({"vector": ["a"]}, weights={"vector": -1.0})
    with pytest.raises(ValueError, match="k must be positive"):
        reciprocal_rank_fusion({"vector": ["a"]}, k=0)


def test_empty_input_is_empty_output() -> None:
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"vector": []}) == []


def test_default_k_matches_the_paper() -> None:
    assert DEFAULT_RRF_K == 60


def test_interleave_keeps_first_occurrence_only() -> None:
    assert interleave(["a", "b"], ["b", "c"]) == ["a", "b", "c"]


def test_duplicates_inside_one_list_are_not_double_counted() -> None:
    fused = rrf_scores({"vector": ["a", "a", "b"]})
    assert fused["a"] == pytest.approx(1 / 61)
