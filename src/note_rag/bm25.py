"""A small, dependency-free BM25 implementation.

This exists for three reasons:

1. It is the *offline* keyword retriever: tests and the eval harness run without Postgres.
2. It gives you a fair baseline to compare Postgres full-text search against.
3. "I implemented BM25 including its failure modes" is a much better interview answer than
   "I called a library".

Tokenizer detail that matters for a Chinese note vault: without a segmentation dictionary we
index CJK **unigrams + bigrams** and keep ASCII words whole. That is a well known cheap
approximation and it keeps recall acceptable for Chinese without pulling in jieba.

Scoring quirk worth knowing before you quote a number: a term repeated in the *query* is
scored once per occurrence, because :meth:`BM25.score` iterates the token list rather than
the token set. Textbook BM25 counts each query term once (or weights by query tf). This is
kept deliberately -- changing it would move every historical number in docs/experiments --
but it is a real deviation, not an accident. See docs/reviews/design-006, section 2.3.
"""

from __future__ import annotations

import bisect
import heapq
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


def content_tokens(text: str) -> list[str]:
    """The "topical" tokens of a query: ASCII words of 4+ chars plus CJK bigrams.

    Short ASCII words (the/is/a) and CJK unigrams occur in almost every note, so leaving
    them in dilutes any coverage measure into noise.

    This lives next to :func:`tokenize` on purpose: the eval-set gate that admits an
    unanswerable question and the runtime signal that decides to abstain must use the same
    definition, or the numbers they produce cannot be compared.
    """
    return sorted(
        {
            token
            for token in tokenize(text)
            if (token.isascii() and len(token) >= 4) or (not token.isascii() and len(token) == 2)
        }
    )


def _rank_key(item: tuple[str, float]) -> tuple[float, str]:
    """Descending score, ties broken by ascending doc id -- eval numbers must be stable."""
    return (-item[1], item[0])


class BM25:
    """Okapi BM25 over an in-memory corpus, backed by an inverted index.

    Args:
        k1: term-frequency saturation. Higher = raw frequency matters more.
        b: length normalisation in ``[0, 1]``. 0 disables it, 1 fully normalises.

    The classic tuning story you should be able to tell: ``k1`` around 1.2-2.0 and ``b``
    around 0.75 are the defaults because they work across corpora; on very short documents
    (like chat/notes) lowering ``b`` tends to help, because length normalisation mostly
    punishes short documents that legitimately contain the query terms.

    **Layout.** Documents live in *slots* (a stable integer index). ``_postings`` maps each
    term to the ascending list of slots containing it, with ``_posting_tfs`` holding the
    matching term frequencies. Scoring walks only those postings, so the inner loop visits
    the ~43k (document, term) pairs that actually match instead of the ~596k pairs a full
    scan tests. Keeping postings ascending is what lets :meth:`score` and :meth:`remove`
    find a slot with ``bisect`` instead of storing a second per-document term map.

    Note that the postings union is essentially the whole corpus here -- terms like "agent"
    occur in every note -- so the win is skipping ``tf == 0`` pairs, not shrinking the
    candidate set. See docs/reviews/design-006-bm25-inverted-index.md.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be within [0, 1]")
        self.k1 = k1
        self.b = b
        self._slots: list[str | None] = []
        self._index_of: dict[str, int] = {}
        self._free: list[int] = []
        self._doc_terms: list[tuple[str, ...]] = []
        self._doc_len: list[int] = []
        self._total_len = 0
        self._live = 0
        self._postings: dict[str, list[int]] = {}
        self._posting_tfs: dict[str, list[int]] = {}
        # Length normalisation depends on avgdl, so every write invalidates it. Rebuilt on
        # the next query: the access pattern is "build once, query many times".
        self._norms: list[float] | None = None

    def __len__(self) -> int:
        return self._live

    @property
    def _avgdl(self) -> float:
        return self._total_len / self._live if self._live else 0.0

    def add(self, doc_id: str, text: str) -> None:
        """Index (or re-index) one document."""
        if doc_id in self._index_of:
            self.remove(doc_id)
        tokens = tokenize(text)
        counts = Counter(tokens)
        slot = self._free.pop() if self._free else -1
        if slot < 0:
            slot = len(self._slots)
            self._slots.append(doc_id)
            self._doc_terms.append(())
            self._doc_len.append(0)
        else:
            self._slots[slot] = doc_id
        self._index_of[doc_id] = slot
        self._doc_terms[slot] = tuple(counts)
        self._doc_len[slot] = len(tokens)
        self._total_len += len(tokens)
        self._live += 1
        for term, tf in counts.items():
            docs = self._postings.get(term)
            if docs is None:
                self._postings[term] = [slot]
                self._posting_tfs[term] = [tf]
            elif slot > docs[-1]:
                # Bulk indexing hands out ascending slots, so this is the hot path.
                docs.append(slot)
                self._posting_tfs[term].append(tf)
            else:
                # A recycled slot: keep the postings ascending so bisect stays valid.
                at = bisect.bisect_left(docs, slot)
                docs.insert(at, slot)
                self._posting_tfs[term].insert(at, tf)
        self._norms = None

    def extend(self, docs: Iterable[tuple[str, str]]) -> None:
        for doc_id, text in docs:
            self.add(doc_id, text)

    def remove(self, doc_id: str) -> None:
        """Drop a document, purging its postings eagerly.

        Eager purging rather than a tombstone: a tombstone would put an "is this slot
        still alive?" test in the scoring inner loop, which is exactly the loop this index
        exists to make cheap. Deletion is the cold path (the eval harness and the API
        rebuild the index wholesale), so the cost belongs here.
        """
        slot = self._index_of.pop(doc_id, None)
        if slot is None:
            return
        for term in self._doc_terms[slot]:
            docs = self._postings[term]
            at = bisect.bisect_left(docs, slot)
            del docs[at]
            del self._posting_tfs[term][at]
            if not docs:
                del self._postings[term]
                del self._posting_tfs[term]
        self._total_len -= self._doc_len[slot]
        self._live -= 1
        self._slots[slot] = None
        self._doc_terms[slot] = ()
        self._doc_len[slot] = 0
        self._free.append(slot)
        self._norms = None

    def _ensure_norms(self) -> list[float]:
        """Per-slot ``k1 * (1 - b + b * dl / avgdl)``, cached until the next write."""
        norms = self._norms
        if norms is None:
            avgdl = self._avgdl or 1.0
            k1, b = self.k1, self.b
            norms = [k1 * (1 - b + b * dl / avgdl) for dl in self._doc_len]
            self._norms = norms
        return norms

    def document_frequency(self, term: str) -> int:
        """How many live documents contain ``term``.

        Read straight off the posting list rather than kept in a parallel counter: a second
        structure holding the same fact is a second structure that can drift out of sync,
        and maintaining it cost ~0.4 s of every index build.
        """
        return len(self._postings.get(term, ()))

    def idf(self, term: str) -> float:
        """BM25 idf. The ``+0.5`` smoothing keeps it non-negative for very common terms."""
        n = self._live
        df = self.document_frequency(term)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_id: str) -> float:
        slot = self._index_of.get(doc_id)
        if slot is None:
            return 0.0
        norm = self._ensure_norms()[slot]
        k1p1 = self.k1 + 1
        total = 0.0
        # Same loop as search(): query-token order, duplicates included (see the module
        # note on repeated query terms), so the float accumulation order matches.
        for term in tokenize(query):
            docs = self._postings.get(term)
            if docs is None:
                continue
            at = bisect.bisect_left(docs, slot)
            if at == len(docs) or docs[at] != slot:
                continue
            tf = self._posting_tfs[term][at]
            total += self.idf(term) * (tf * k1p1) / (tf + norm)
        return total

    def search(self, query: str, *, limit: int = 10) -> list[tuple[str, float]]:
        """Rank all documents; returns ``[(doc_id, score), ...]`` descending.

        Empty-query tokens or all-zero scores yield an empty list rather than noise.
        """
        q_tokens = tokenize(query)
        if not q_tokens or not self._live:
            return []
        # Hoisted out of the per-document loop. The previous version called tokenize() and
        # idf() once per *document*, which was ~83% of query time at 17.5k chunks.
        idf = {term: self.idf(term) for term in set(q_tokens)}
        norms = self._ensure_norms()
        slots = self._slots
        scores = [0.0] * len(slots)
        k1p1 = self.k1 + 1
        for term in q_tokens:
            docs = self._postings.get(term)
            if docs is None:
                continue
            weight = idf[term]
            tfs = self._posting_tfs[term]
            for at, slot in enumerate(docs):
                tf = tfs[at]
                scores[slot] += weight * (tf * k1p1) / (tf + norms[slot])
        ranked = [
            (doc_id, score)
            for doc_id, score in zip(slots, scores, strict=True)
            if score > 0.0 and doc_id is not None
        ]
        key = _rank_key
        if limit < 0:
            ranked.sort(key=key)
            return ranked
        # nsmallest is documented as sorted(iterable, key=key)[:n]; doc ids are unique so
        # the key is a total order and there are no ties whose stability could differ.
        return heapq.nsmallest(limit, ranked, key=key)

    def top_terms(self, query: str, limit: int = 20) -> Sequence[str]:
        """Query terms sortable by idf -- handy when explaining why a query failed."""
        terms = sorted(set(tokenize(query)), key=self.idf, reverse=True)
        return terms[:limit]
