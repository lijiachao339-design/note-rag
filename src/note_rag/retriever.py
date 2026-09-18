"""The retrieval pipeline: one module, four modes, no duplicated logic.

Modes exist so the eval harness can run an ablation from a single implementation:

* ``vector``   -- embedding search only (pgvector)
* ``keyword``  -- BM25 only (in-process)
* ``hybrid``   -- RRF fusion of the two
* ``hybrid_rerank`` -- hybrid, then a cross-encoder rerank of the top candidates

The reranker is deliberately pluggable and defaults to a no-op, so the ablation table shows
"hybrid vs hybrid+rerank" honestly rather than pretending a reranker is present.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from note_rag import store
from note_rag.bm25 import BM25
from note_rag.config import Settings
from note_rag.embed import Embedder, build_embedder
from note_rag.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion

Mode = Literal["vector", "keyword", "hybrid", "hybrid_rerank"]


@dataclass(slots=True, frozen=True)
class SearchHit:
    chunk_id: str
    note_path: str
    title: str
    heading_path: tuple[str, ...]
    text: str
    score: float


class Reranker:
    """Cross-encoder reranker interface. Return the candidates reordered (best first).

    Implementation sketch: load ``bge-reranker-base`` with ``sentence-transformers`` on CPU,
    score ``(query, chunk_text)`` pairs, sort by score. On a machine without CUDA, limit the
    candidate count (<=50) and cache scores by ``(query, chunk_id)``.
    """

    name = "noop"

    def rerank(self, query: str, candidates: Sequence[SearchHit]) -> list[SearchHit]:
        return list(candidates)


class Retriever:
    """Owns the vector index (Postgres) and the keyword index (in-process BM25)."""

    def __init__(
        self,
        *,
        conn: object,
        embedder: Embedder,
        bm25: BM25 | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_k: int = 50,
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
        reranker: Reranker | None = None,
    ) -> None:
        self.conn = conn
        self.embedder = embedder
        self.bm25 = bm25 if bm25 is not None else BM25()
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.weights = {"vector": vector_weight, "keyword": keyword_weight}
        self.reranker = reranker or Reranker()
        self._chunks: dict[str, SearchHit] = {}

    # -- construction ---------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Settings) -> Retriever:
        conn = store.connect(settings.database_url)
        # 显式设置 HNSW 的 ef_search：默认 40 小于 candidate_k(50)，会让向量这一路白掉召回。
        store.set_hnsw_ef_search(conn, settings.hnsw_ef_search)
        return cls(
            conn=conn,
            embedder=build_embedder(
                base_url=settings.embed_base_url,
                api_key=settings.embed_api_key,
                model=settings.embed_model,
                dim=settings.embed_dim,
                batch_size=settings.embed_batch_size,
            ),
            rrf_k=settings.rrf_k,
            candidate_k=settings.candidate_k,
            vector_weight=settings.vector_weight,
            keyword_weight=settings.keyword_weight,
        )

    def build_keyword_index(self) -> int:
        """Load chunks from Postgres into BM25 and into the result cache.

        Runs at startup. At 10k chunks this is a few hundred milliseconds; if the corpus
        grows, move it behind a background refresh and serve stale reads meanwhile.
        """
        rows = store.load_chunk_rows(self.conn)  # type: ignore[arg-type]
        self._chunks = {
            row.id: SearchHit(
                chunk_id=row.id,
                note_path=row.note_path,
                title=row.title,
                heading_path=row.heading_path,
                text=row.text,
                score=0.0,
            )
            for row in rows
        }
        self.bm25 = BM25()
        self.bm25.extend((row.id, f"{row.title} {row.text}") for row in rows)
        return len(rows)

    # -- introspection --------------------------------------------------------------

    def corpus_stats(self) -> dict[str, int]:
        return store.corpus_stats(self.conn)  # type: ignore[arg-type]

    def index_stats(self) -> dict[str, int]:
        return {"keyword_index_docs": len(self.bm25), "chunks_cached": len(self._chunks)}

    def indexed_notes(self) -> set[str]:
        """Every note path currently in the index (builds the index on first use).

        The eval harness calls this to fail loudly when a labelled note was never indexed.
        Without it, a note the indexer skipped simply scores 0 for every mode and every
        query that points at it -- indistinguishable from "retrieval is bad". See
        docs/reviews/review-001, S10.
        """
        self._ensure_index()
        return {hit.note_path for hit in self._chunks.values() if hit.note_path}

    # -- retrieval ------------------------------------------------------------------

    def _vector_ranking(self, query: str, limit: int) -> list[str]:
        [vector] = self.embedder.embed([query])
        return [chunk_id for chunk_id, _ in store.vector_search(self.conn, vector, limit=limit)]  # type: ignore[arg-type]

    def _keyword_ranking(self, query: str, limit: int) -> list[str]:
        return [chunk_id for chunk_id, _ in self.bm25.search(query, limit=limit)]

    def _ensure_index(self) -> None:
        """Build the keyword index and hydration cache on first use.

        Regression note: the API server used to create the retriever without ever calling
        :meth:`build_keyword_index`, and because :meth:`_hydrate` degrades gracefully for
        unknown chunk ids, every hit came back with an empty title/text/note_path while
        ``/healthz`` and ``/metrics`` stayed green. Making ``search`` self-sufficient
        removes that entire class of failure.
        """
        if not self._chunks:
            self.build_keyword_index()

    def _hydrate(self, chunk_id: str, score: float) -> SearchHit:
        """Attach stored metadata to a ranked id.

        Unknown ids degrade to an empty hit instead of raising: during a concurrent
        re-ingest a chunk can legitimately disappear between ranking and hydration.
        """
        base = self._chunks.get(chunk_id)
        if base is not None:
            return SearchHit(
                chunk_id=base.chunk_id,
                note_path=base.note_path,
                title=base.title,
                heading_path=base.heading_path,
                text=base.text,
                score=score,
            )
        return SearchHit(
            chunk_id=chunk_id,
            note_path="",
            title="",
            heading_path=(),
            text="",
            score=score,
        )

    def search(self, query: str, *, k: int = 10, mode: Mode = "hybrid") -> list[SearchHit]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if k <= 0:
            raise ValueError("k must be positive")
        self._ensure_index()
        candidate_k = max(k, self.candidate_k)

        if mode == "vector":
            ranked = self._vector_ranking(query, candidate_k)[:k]
            return [self._hydrate(cid, float(candidate_k - i)) for i, cid in enumerate(ranked)]

        if mode == "keyword":
            weighted = self.bm25.search(query, limit=candidate_k)[:k]
            return [self._hydrate(cid, score) for cid, score in weighted]

        rankings = {
            "vector": self._vector_ranking(query, candidate_k),
            "keyword": self._keyword_ranking(query, candidate_k),
        }
        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k, weights=self.weights)
        hits = [self._hydrate(cid, score) for cid, score in fused[:candidate_k]]

        if mode == "hybrid_rerank":
            hits = self.reranker.rerank(query, hits)
        return hits[:k]

    def search_notes(self, query: str, *, k: int = 5, mode: Mode = "hybrid") -> list[SearchHit]:
        """Retrieval at *note* granularity: the best-scoring chunk per note, deduplicated.

        Note-level ranking is what the eval harness scores, because labelling relevant
        *chunks* by hand does not scale and note-level labels are what a human actually
        judges ("this note answers my question").
        """
        chunks = self.search(query, k=k * 4, mode=mode)
        seen: dict[str, SearchHit] = {}
        for hit in chunks:
            if hit.note_path and hit.note_path not in seen:
                seen[hit.note_path] = hit
        return list(seen.values())[:k]
