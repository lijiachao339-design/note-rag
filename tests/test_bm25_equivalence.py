"""Bit-for-bit equivalence between the inverted index and the full-scan BM25 it replaced.

The pre-inverted-index implementation is inlined below as a reference scorer rather than
imported, so it cannot drift with later refactors. Scores must match *exactly*: the design
(docs/reviews/design-006, section 2.3) argues the float accumulation order is unchanged, so
any tolerance here would be hiding a real difference rather than absorbing rounding.
"""

from __future__ import annotations

import math
import random

from note_rag.bm25 import BM25, tokenize

VOCAB_EN = ["retrieval", "vector", "bm25", "rrf", "index", "note", "agent", "chunk", "score"]
VOCAB_ZH = ["混合检索", "向量索引", "评测指标", "笔记", "融合排名", "长度归一化"]


class ReferenceBM25:
    """The previous implementation, verbatim: score every document, tokenize per document."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_ids: list[str] = []
        self._doc_tokens: dict[str, dict[str, int]] = {}
        self._doc_len: dict[str, int] = {}
        self._df: dict[str, int] = {}
        self._avgdl = 0.0

    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        self._doc_ids.append(doc_id)
        self._doc_tokens[doc_id] = counts
        self._doc_len[doc_id] = len(tokens)
        for term in counts:
            self._df[term] = self._df.get(term, 0) + 1
        self._avgdl = sum(self._doc_len.values()) / len(self._doc_len)

    def idf(self, term: str) -> float:
        n = len(self._doc_ids)
        df = self._df.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_id: str) -> float:
        counts = self._doc_tokens.get(doc_id)
        if not counts:
            return 0.0
        dl = self._doc_len[doc_id]
        avgdl = self._avgdl or 1.0
        total = 0.0
        for term in tokenize(query):
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
            total += self.idf(term) * (tf * (self.k1 + 1)) / denom
        return total

    def search(self, query: str, *, limit: int = 10) -> list[tuple[str, float]]:
        scored = [(doc_id, self.score(query, doc_id)) for doc_id in self._doc_ids]
        ranked = [(doc_id, score) for doc_id, score in scored if score > 0.0]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:limit] if limit >= 0 else ranked


def _corpus(seed: int, docs: int = 120) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    out = []
    for i in range(docs):
        words = [rng.choice(VOCAB_EN) for _ in range(rng.randint(3, 60))]
        words += [rng.choice(VOCAB_ZH) for _ in range(rng.randint(0, 20))]
        rng.shuffle(words)
        out.append((f"d{i:04d}", " ".join(words)))
    return out


def _queries(seed: int, n: int = 60) -> list[str]:
    rng = random.Random(seed + 1)
    out = []
    for _ in range(n):
        terms = [rng.choice(VOCAB_EN + VOCAB_ZH) for _ in range(rng.randint(1, 6))]
        if rng.random() < 0.4:  # deliberately repeat a term: the quirk must survive
            terms.append(terms[0])
        out.append(" ".join(terms))
    return out


def test_search_matches_the_full_scan_implementation_bit_for_bit() -> None:
    docs = _corpus(7)
    new, old = BM25(), ReferenceBM25()
    new.extend(docs)
    for doc_id, text in docs:
        old.add(doc_id, text)

    for query in _queries(7):
        assert new.search(query, limit=-1) == old.search(query, limit=-1), query


def test_top_k_slice_matches_too() -> None:
    """nsmallest(limit) must agree with sorted(...)[:limit], including the tie-break."""
    docs = _corpus(11)
    new, old = BM25(), ReferenceBM25()
    new.extend(docs)
    for doc_id, text in docs:
        old.add(doc_id, text)

    for query in _queries(11):
        for limit in (1, 5, 50):
            assert new.search(query, limit=limit) == old.search(query, limit=limit)


def test_score_of_one_document_matches_the_reference() -> None:
    docs = _corpus(13)
    new, old = BM25(), ReferenceBM25()
    new.extend(docs)
    for doc_id, text in docs:
        old.add(doc_id, text)

    for query in _queries(13)[:20]:
        for doc_id, _ in docs[:15]:
            assert new.score(query, doc_id) == old.score(query, doc_id)


def test_equivalence_survives_removals_and_reindexing() -> None:
    """Slot recycling must not disturb scores -- postings have to stay ascending."""
    docs = _corpus(17, docs=60)
    new, old = BM25(), ReferenceBM25()
    new.extend(docs)

    rng = random.Random(17)
    dropped = {doc_id for doc_id, _ in rng.sample(docs, 12)}
    replaced = dict(rng.sample([(d, t) for d, t in docs if d not in dropped], 8))
    for doc_id in dropped:
        new.remove(doc_id)
    for doc_id in replaced:
        new.add(doc_id, "bm25 融合排名 chunk chunk")

    # Rebuild the reference from the resulting corpus, which is what "unchanged semantics" means.
    for doc_id, text in docs:
        if doc_id in dropped:
            continue
        old.add(doc_id, "bm25 融合排名 chunk chunk" if doc_id in replaced else text)

    assert len(new) == len(docs) - len(dropped)
    for query in _queries(17)[:30]:
        assert new.search(query, limit=-1) == old.search(query, limit=-1), query
