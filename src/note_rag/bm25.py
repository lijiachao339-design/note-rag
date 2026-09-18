"""A small, dependency-free BM25 implementation.

This exists for three reasons:

1. It is the *offline* keyword retriever: tests and the eval harness run without Postgres.
2. It gives you a fair baseline to compare Postgres full-text search against.
3. "I implemented BM25 including its failure modes" is a much better interview answer than
   "I called a library".

Tokenizer detail that matters for a Chinese note vault: without a segmentation dictionary we
index CJK **unigrams + bigrams** and keep ASCII words whole. That is a well known cheap
approximation and it keeps recall acceptable for Chinese without pulling in jieba.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

_ASCII_WORD = re.compile(r"[a-z0-9]+(?:[._'-][a-z0-9]+)*")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Lowercase, then emit ASCII words plus CJK unigrams and bigrams."""
    lowered = text.lower()
    tokens = _ASCII_WORD.findall(lowered)

    cjk_runs: list[str] = []
    current: list[str] = []
    for char in lowered:
        if _CJK.match(char):
            current.append(char)
        elif current:
            cjk_runs.append("".join(current))
            current = []
    if current:
        cjk_runs.append("".join(current))

    for run in cjk_runs:
        tokens.extend(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


class BM25:
    """Okapi BM25 over an in-memory corpus.

    Args:
        k1: term-frequency saturation. Higher = raw frequency matters more.
        b: length normalisation in ``[0, 1]``. 0 disables it, 1 fully normalises.

    The classic tuning story you should be able to tell: ``k1`` around 1.2-2.0 and ``b``
    around 0.75 are the defaults because they work across corpora; on very short documents
    (like chat/notes) lowering ``b`` tends to help, because length normalisation mostly
    punishes short documents that legitimately contain the query terms.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be within [0, 1]")
        self.k1 = k1
        self.b = b
        self._doc_ids: list[str] = []
        self._doc_tokens: dict[str, Counter[str]] = {}
        self._doc_len: dict[str, int] = {}
        self._df: Counter[str] = Counter()
        self._avgdl: float = 0.0

    def __len__(self) -> int:
        return len(self._doc_ids)

    def add(self, doc_id: str, text: str) -> None:
        """Index (or re-index) one document."""
        if doc_id in self._doc_tokens:
            self.remove(doc_id)
        tokens = tokenize(text)
        counts = Counter(tokens)
        self._doc_ids.append(doc_id)
        self._doc_tokens[doc_id] = counts
        self._doc_len[doc_id] = len(tokens)
        self._df.update(counts.keys())
        self._recompute_avgdl()

    def extend(self, docs: Iterable[tuple[str, str]]) -> None:
        for doc_id, text in docs:
            self.add(doc_id, text)

    def remove(self, doc_id: str) -> None:
        counts = self._doc_tokens.pop(doc_id, None)
        if counts is None:
            return
        self._doc_ids.remove(doc_id)
        self._doc_len.pop(doc_id, None)
        self._df.subtract(counts.keys())
        self._df = Counter({term: df for term, df in self._df.items() if df > 0})
        self._recompute_avgdl()

    def _recompute_avgdl(self) -> None:
        self._avgdl = sum(self._doc_len.values()) / len(self._doc_len) if self._doc_len else 0.0

    def idf(self, term: str) -> float:
        """BM25 idf. The ``+0.5`` smoothing keeps it non-negative for very common terms."""
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
        """Rank all documents; returns ``[(doc_id, score), ...]`` descending.

        Empty-query tokens or all-zero scores yield an empty list rather than noise.
        """
        scored = [(doc_id, self.score(query, doc_id)) for doc_id in self._doc_ids]
        ranked = [(doc_id, score) for doc_id, score in scored if score > 0.0]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:limit] if limit >= 0 else ranked

    def top_terms(self, query: str, limit: int = 20) -> Sequence[str]:
        """Query terms sortable by idf -- handy when explaining why a query failed."""
        terms = sorted(set(tokenize(query)), key=self.idf, reverse=True)
        return terms[:limit]
