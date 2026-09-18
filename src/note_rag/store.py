"""Postgres + pgvector storage layer.

Deliberate design choices worth defending in an interview:

* **Vectors live in Postgres**, not in a dedicated vector DB. At ~10k chunks, HNSW in
  pgvector is comfortably fast, and you get transactions, filtering, and one backup story
  for free. Reaching for Milvus/Chroma here would be resume-driven engineering.
* **Keyword retrieval runs in-process** (see :mod:`note_rag.bm25`) over chunks loaded once
  at startup. Postgres FTS with the ``simple`` dictionary does not tokenise Chinese without
  ``zhparser``/``pg_bigm``, and an honest in-process BM25 is both more portable and more
  explainable. Experiment: implement pg FTS with ``pg_bigm`` and report the delta.
* **``distance`` is cosine distance** (``<=>`` for pgvector), and similarity is reported as
  ``1 - distance`` so callers never have to remember which direction is "better".
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import psycopg

from note_rag.notes import Chunk

SCHEMA_TEMPLATE = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id            text PRIMARY KEY,
    note_path     text        NOT NULL,
    title         text        NOT NULL,
    heading_path  text        NOT NULL,
    text          text        NOT NULL,
    content_hash  text        NOT NULL,
    char_start    integer     NOT NULL,
    char_end      integer     NOT NULL,
    embedding     vector({dim}) NOT NULL,
    updated_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_note_path_idx ON chunks (note_path);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""


@dataclass(slots=True, frozen=True)
class ChunkRow:
    id: str
    note_path: str
    title: str
    heading_path: tuple[str, ...]
    text: str


def connect(database_url: str) -> psycopg.Connection:
    """Open a connection with autocommit off (explicit transactions)."""
    return psycopg.connect(database_url)


def _as_vector_literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in vec) + "]"


def ensure_schema(conn: psycopg.Connection, *, dim: int = 384) -> None:
    """Create the extension, table and indexes. Safe to run repeatedly."""
    if dim <= 0:
        raise ValueError("dim must be positive")
    with conn.cursor() as cur:
        cur.execute(SCHEMA_TEMPLATE.format(dim=int(dim)))
    conn.commit()


def upsert_chunks(
    conn: psycopg.Connection,
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
    *,
    embedder_name: str = "",
) -> int:
    """Insert or update chunks. Returns the number of rows written.

    ``ON CONFLICT`` on the primary key makes ingestion idempotent, which is what lets an
    interrupted ingest simply be re-run instead of requiring a full rebuild.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")
    if not chunks:
        return 0
    dim = len(embeddings[0])
    for vec in embeddings:
        if len(vec) != dim:
            raise ValueError("all embeddings must share one dimension")

    rows = [
        (
            chunk.id,
            chunk.note_path,
            chunk.title,
            " > ".join(chunk.heading_path),
            chunk.text,
            chunk.content_hash,
            chunk.char_start,
            chunk.char_end,
            _as_vector_literal(vec),
        )
        for chunk, vec in zip(chunks, embeddings, strict=True)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (id, note_path, title, heading_path, text, content_hash,
                                char_start, char_end, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
            ON CONFLICT (id) DO UPDATE SET
                text         = EXCLUDED.text,
                title        = EXCLUDED.title,
                heading_path = EXCLUDED.heading_path,
                content_hash = EXCLUDED.content_hash,
                char_start   = EXCLUDED.char_start,
                char_end     = EXCLUDED.char_end,
                embedding    = EXCLUDED.embedding,
                updated_at   = now()
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def delete_chunks(conn: psycopg.Connection, ids: Iterable[str]) -> int:
    ids = list(ids)
    if not ids:
        return 0
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE id = ANY(%s)", (ids,))
        deleted = cur.rowcount
    conn.commit()
    return deleted


def existing_hashes(conn: psycopg.Connection) -> dict[str, str]:
    """``{chunk_id: content_hash}`` for the whole corpus.

    This is the incremental-indexing primitive: only chunks whose hash changed (or that are
    new) need an embedding call.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT id, content_hash FROM chunks")
        return {row[0]: row[1] for row in cur.fetchall()}


def load_chunk_rows(conn: psycopg.Connection) -> list[ChunkRow]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, note_path, title, heading_path, text FROM chunks")
        return [
            ChunkRow(
                id=row[0],
                note_path=row[1],
                title=row[2],
                heading_path=tuple(row[3].split(" > ")) if row[3] else (),
                text=row[4],
            )
            for row in cur.fetchall()
        ]


def vector_search(
    conn: psycopg.Connection,
    embedding: Sequence[float],
    *,
    limit: int = 50,
) -> list[tuple[str, float]]:
    """Cosine similarity search over chunk embeddings."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, 1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (_as_vector_literal(embedding), _as_vector_literal(embedding), limit),
        )
        return [(row[0], float(row[1])) for row in cur.fetchall()]


def corpus_stats(conn: psycopg.Connection) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*), count(DISTINCT note_path) FROM chunks")
        chunks, notes = cur.fetchone() or (0, 0)
    return {"chunks": int(chunks), "notes": int(notes)}


def note_paths_for(conn: psycopg.Connection, chunk_ids: Sequence[str]) -> dict[str, str]:
    """``{chunk_id: note_path}`` -- used to score retrieval at the note level."""
    if not chunk_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute("SELECT id, note_path FROM chunks WHERE id = ANY(%s)", (list(chunk_ids),))
        return {row[0]: row[1] for row in cur.fetchall()}
