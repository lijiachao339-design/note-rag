"""Reciprocal Rank Fusion (RRF) and friends.

Why RRF instead of a weighted sum of scores:

* Vector similarity (cosine, roughly ``[0, 1]``) and BM25 (unbounded, corpus dependent)
  are **not** on a comparable scale, so a weighted sum needs per-corpus calibration and
  silently breaks when the corpus changes. RRF only consumes *ranks*, so it is scale free.
* It is one line of math, has no training data, and is embarrassingly robust -- which is
  exactly what you want for a hybrid retriever's first stage.

Reference: Cormack, Clarke & Buettcher (2009), "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse several ranked id lists into one.

    Args:
        rankings: ``{retriever_name: [doc_id, ...]}`` ordered best-first.
        k: RRF smoothing constant (60 is the value from the paper; smaller values
            weight the very top of each list more aggressively).
        weights: optional per-retriever multiplier, e.g. ``{"vector": 1.0, "bm25": 0.7}``.

    Returns:
        ``[(doc_id, score), ...]`` sorted by descending score. Ties are broken
        deterministically by best rank, then by id, so results are reproducible --
        a non-deterministic ranker makes eval numbers meaningless. Duplicate ids *within*
        one ranked list are counted once, at their best rank.

    Raises:
        ValueError: if ``k`` is not positive or a weight is negative.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}

    for name, doc_ids in rankings.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        if weight < 0:
            raise ValueError(f"weight for {name!r} must be non-negative")
        if weight == 0:
            continue
        seen: set[str] = set()
        for rank, doc_id in enumerate(doc_ids, start=1):
            # A ranked list containing duplicates is an upstream bug, and a repeated id must
            # not be able to buy itself extra score. Keep only the first (best) occurrence.
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
            if doc_id not in best_rank or rank < best_rank[doc_id]:
                best_rank[doc_id] = rank

    return sorted(scores.items(), key=lambda item: (-item[1], best_rank[item[0]], item[0]))


def rrf_scores(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Same as :func:`reciprocal_rank_fusion` but keyed by doc id."""
    return dict(reciprocal_rank_fusion(rankings, k=k, weights=weights))


def interleave(*rankings: Sequence[str]) -> list[str]:
    """Round-robin merge, keeping first occurrence. Useful as a trivial baseline in eval.

    A baseline that takes one line to write is worth keeping: it is what turns
    "RRF feels better" into "RRF beats round-robin by +0.12 recall@5".
    """
    seen: set[str] = set()
    out: list[str] = []
    for group in zip(*rankings, strict=False):
        for doc_id in group:
            if doc_id not in seen:
                seen.add(doc_id)
                out.append(doc_id)
    return out
