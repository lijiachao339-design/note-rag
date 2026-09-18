"""Retrieval metrics: recall@k, MRR, nDCG@k, plus a report builder.

This module is the reason the project has numbers on its resume. Rules that keep the
numbers honest:

* Graded relevance is supported (``qrels[qid] = {doc_id: grade}``); a plain ``set`` means
  every relevant document has grade 1.
* Unanswerable questions (a qid whose ``qrels`` entry is empty) score 0 on every ranking
  metric **no matter what the retriever returned**. That makes them useless here: a
  retriever that correctly returns nothing and one that invents ten hits get the same 0.
  So they are *excluded* from the ranking report and measured separately by
  :func:`abstention_report`, which is the function that actually catches a retriever that
  hallucinates relevance. (An earlier version of this docstring claimed the opposite; the
  code never supported it. See docs/reviews/review-001, S4.)
* ``precision@k`` is deliberately absent from :func:`evaluate`: with a fixed number of
  relevant documents per query it is ``recall@k * |relevant| / k``, a constant multiple that
  adds a column and no information (docs/reviews/review-001, S6).
* Every function returns plain floats so the report can be diffed between git commits.
"""

from __future__ import annotations

import math
import random
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

    Naming caveat worth stating out loud: when a query has exactly **one** relevant
    document this is 0 or 1, i.e. it is *Success@k* / *Hit Rate@k*, not what an IR
    textbook means by recall. This dataset has two relevant notes for most queries (a note
    and its translated twin), so it is a real recall -- which also means it is **not
    comparable** to the single-label numbers in exp-001. See docs/reviews/review-001, S6.

    Returns 0.0 when there is nothing relevant to find. Callers must not average that in:
    it is a constant, not a measurement (see the module docstring).
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
        report[f"ndcg@{k}"] = sum(ndcg_at_k(rankings[qid], qrels[qid], k) for qid in rankings) / n
    report["mrr"] = sum(reciprocal_rank(rankings[qid], qrels[qid]) for qid in rankings) / n
    return report


def abstention_report(
    abstained: Mapping[str, bool], answerable: Mapping[str, bool]
) -> dict[str, float]:
    """How well a retriever knows when to say nothing.

    This is the measurement the unanswerable questions exist for. The ranking metrics
    cannot express it: with no relevant document, recall/MRR/nDCG are 0 for every possible
    behaviour, so "returned nothing" and "invented ten hits" are indistinguishable there.

    Returns ``abstain_rate_on_unanswerable`` (higher is better) and
    ``false_abstain_rate_on_answerable`` (the price). Report them as a pair: abstaining on
    everything scores a perfect 1.0 on the first and is obviously useless.
    """
    missing = [qid for qid in abstained if qid not in answerable]
    if missing:
        raise ValueError(f"answerable flag missing for {len(missing)} queries, e.g. {missing[:3]}")

    unanswerable_ids = [qid for qid in abstained if not answerable[qid]]
    answerable_ids = [qid for qid in abstained if answerable[qid]]
    report: dict[str, float] = {
        "unanswerable": float(len(unanswerable_ids)),
        "answerable": float(len(answerable_ids)),
    }
    if unanswerable_ids:
        report["abstain_rate_on_unanswerable"] = sum(
            1.0 for qid in unanswerable_ids if abstained[qid]
        ) / len(unanswerable_ids)
    if answerable_ids:
        report["false_abstain_rate_on_answerable"] = sum(
            1.0 for qid in answerable_ids if abstained[qid]
        ) / len(answerable_ids)
    return report


def cluster_bootstrap_ci(
    per_query: Mapping[str, float],
    cluster_of: Mapping[str, str],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``per_query``, resampling **clusters**.

    The resampling unit has to be the cluster, not the query: this dataset draws 3 questions
    from each source note, and whether a note is retrieved is shared by all 3. Treating them
    as independent understates the variance -- measured at ~23% too narrow on this set
    (docs/reviews/review-001, S7).

    ``seed`` is fixed so the interval is reproducible; a CI that moves between runs is not
    something you can put in an experiment document.
    """
    if not per_query:
        raise ValueError("per_query must not be empty")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be within (0, 1)")
    missing = [qid for qid in per_query if qid not in cluster_of]
    if missing:
        raise ValueError(f"cluster missing for {len(missing)} queries, e.g. {missing[:3]}")

    grouped: dict[str, list[float]] = {}
    for qid, value in per_query.items():
        grouped.setdefault(cluster_of[qid], []).append(value)
    clusters = sorted(grouped)

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        drawn: list[float] = []
        for _ in clusters:
            drawn.extend(grouped[clusters[rng.randrange(len(clusters))]])
        means.append(sum(drawn) / len(drawn))
    means.sort()
    tail = (1.0 - confidence) / 2.0
    lo = means[min(len(means) - 1, int(tail * samples))]
    hi = means[min(len(means) - 1, int((1.0 - tail) * samples))]
    return (lo, hi)


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
