"""Retrieval metrics: recall@k, MRR, nDCG@k, plus a report builder.

This module is the reason the project has numbers on its resume. Rules that keep the
numbers honest:

* Graded relevance is supported (``qrels[qid] = {doc_id: grade}``); a plain ``set`` means
  every relevant document has grade 1.
* Unanswerable questions are allowed. Put a qid in ``qrels`` with no relevant documents and
  it is scored as *not* retrieved but *counted* -- that is how you catch a retriever that
  hallucinates relevance.
* Every function returns plain floats so the report can be diffed between git commits.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

Qrels = Mapping[str, Mapping[str, float] | Iterable[str]]
Rankings = Mapping[str, Sequence[str]]


def _as_grades(relevant: Mapping[str, float] | Iterable[str]) -> dict[str, float]:
    if isinstance(relevant, Mapping):
        return {str(doc): float(grade) for doc, grade in relevant.items()}
    return {str(doc): 1.0 for doc in relevant}


def recall_at_k(
    ranked: Sequence[str], relevant: Mapping[str, float] | Iterable[str], k: int
) -> float:
    """Fraction of relevant documents found in the top ``k``.

    Returns 0.0 when there is nothing relevant to find (see module docstring: that keeps
    unanswerable questions in the average instead of silently dropping them).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    grades = _as_grades(relevant)
    if not grades:
        return 0.0
    hit = sum(1 for doc in ranked[:k] if doc in grades)
    return hit / len(grades)


def precision_at_k(
    ranked: Sequence[str], relevant: Mapping[str, float] | Iterable[str], k: int
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    grades = _as_grades(relevant)
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for doc in top if doc in grades) / len(top)


def reciprocal_rank(ranked: Sequence[str], relevant: Mapping[str, float] | Iterable[str]) -> float:
    """1 / rank of the first relevant document; 0.0 if none is retrieved."""
    grades = _as_grades(relevant)
    if not grades:
        return 0.0
    for rank, doc in enumerate(ranked, start=1):
        if doc in grades:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked: Sequence[str], relevant: Mapping[str, float] | Iterable[str], k: int
) -> float:
    """Normalised discounted cumulative gain with exponential gain (``2^g - 1``).

    Use this when relevance is graded (e.g. 2 = answers the question, 1 = partially useful).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    grades = _as_grades(relevant)

    dcg = 0.0
    for rank, doc in enumerate(ranked[:k], start=1):
        grade = grades.get(doc, 0.0)
        if grade > 0:
            dcg += (2.0**grade - 1.0) / math.log2(rank + 1)

    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(
        (2.0**grade - 1.0) / math.log2(rank + 1) for rank, grade in enumerate(ideal, start=1)
    )
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(
    rankings: Rankings, qrels: Qrels, *, ks: Sequence[int] = (1, 5, 10)
) -> dict[str, float]:
    """Aggregate metrics over a whole query set.

    The returned mapping is flat (``{"recall@5": 0.87, "mrr": 0.79, ...}``) so it can be
    dumped to JSON and compared across experiments with a two-line script.
    """
    if not rankings:
        raise ValueError("rankings must not be empty")
    missing = [qid for qid in rankings if qid not in qrels]
    if missing:
        raise ValueError(f"qrels missing for {len(missing)} queries, e.g. {missing[:3]}")

    n = len(rankings)
    report: dict[str, float] = {"queries": float(n)}
    for k in ks:
        report[f"recall@{k}"] = (
            sum(recall_at_k(rankings[qid], qrels[qid], k) for qid in rankings) / n
        )
        report[f"precision@{k}"] = (
            sum(precision_at_k(rankings[qid], qrels[qid], k) for qid in rankings) / n
        )
        report[f"ndcg@{k}"] = sum(ndcg_at_k(rankings[qid], qrels[qid], k) for qid in rankings) / n
    report["mrr"] = sum(reciprocal_rank(rankings[qid], qrels[qid]) for qid in rankings) / n
    return report


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile (``p`` in ``[0, 100]``). Used for p50/p95 latency."""
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= p <= 100.0:
        raise ValueError("p must be within [0, 100]")
    ordered = sorted(values)
    index = max(0, math.ceil(p / 100.0 * len(ordered)) - 1)
    return ordered[index]


def format_report(report: Mapping[str, float], *, title: str = "") -> str:
    """Markdown table row block, ready to paste into README or Obsidian."""
    lines = [
        f"### {title}" if title else "### Retrieval report",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key, value in report.items():
        lines.append(
            f"| {key} | {value:.4f} |" if key != "queries" else f"| {key} | {int(value)} |"
        )
    return "\n".join(lines) + "\n"
