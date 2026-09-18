"""Vault ingestion: parse -> chunk -> diff -> embed -> upsert.

The interesting part is the diff. Naively re-embedding a 1,200 note vault on every sync
means thousands of redundant API calls, which costs real money and takes minutes. Here:

1. every chunk carries a stable id derived from ``(note_path, heading_path, char_start)``;
2. Postgres already holds ``{chunk_id: content_hash}``;
3. only chunks that are new **or whose text hash changed** are sent to the embedder;
4. chunk ids that disappeared are deleted (handles deleted notes and edited headings).

Re-running an unchanged sync must therefore report ``embeddings_computed == 0``. That
invariant is asserted in the test suite and is the number you quote on your resume.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from note_rag import store
from note_rag.config import Settings
from note_rag.embed import Embedder
from note_rag.notes import Chunk, chunk_note, iter_vault_files, parse_note


@dataclass(slots=True)
class SyncStats:
    files_scanned: int = 0
    notes_failed: int = 0
    chunks_total: int = 0
    chunks_new: int = 0
    chunks_changed: int = 0
    chunks_unchanged: int = 0
    chunks_deleted: int = 0
    embeddings_computed: int = 0
    elapsed_s: float = 0.0
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "files_scanned": self.files_scanned,
            "notes_failed": self.notes_failed,
            "chunks_total": self.chunks_total,
            "chunks_new": self.chunks_new,
            "chunks_changed": self.chunks_changed,
            "chunks_unchanged": self.chunks_unchanged,
            "chunks_deleted": self.chunks_deleted,
            "embeddings_computed": self.embeddings_computed,
            "elapsed_s": round(self.elapsed_s, 3),
            "failures": self.failures[:10],
        }

    def summary_line(self) -> str:
        return (
            f"{self.chunks_total} chunks from {self.files_scanned} files "
            f"({self.chunks_new} new, {self.chunks_changed} changed, "
            f"{self.chunks_deleted} deleted) -> {self.embeddings_computed} embeddings "
            f"in {self.elapsed_s:.2f}s"
        )


def collect_chunks(
    vault: Path, *, max_chars: int, overlap_chars: int, stats: SyncStats
) -> list[Chunk]:
    """Parse and chunk every markdown file in the vault.

    A single unreadable/broken file must not abort the whole ingest: failures are collected
    and reported, because a vault edited daily will always contain something malformed.
    """
    all_chunks: list[Chunk] = []
    for path in iter_vault_files(vault):
        stats.files_scanned += 1
        rel_path = path.relative_to(vault).as_posix()
        try:
            raw = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            stats.notes_failed += 1
            stats.failures.append(f"{rel_path}: {exc}")
            continue
        try:
            note = parse_note(rel_path, raw)
            all_chunks.extend(chunk_note(note, max_chars=max_chars, overlap_chars=overlap_chars))
        except Exception as exc:
            stats.notes_failed += 1
            stats.failures.append(f"{rel_path}: {type(exc).__name__}: {exc}")
    return all_chunks


def sync_vault(
    *,
    settings: Settings,
    conn: object,
    embedder: Embedder,
    full: bool = False,
    progress: Callable[[str], None] | None = None,
) -> SyncStats:
    """Incrementally sync the vault into Postgres. Returns ingestion statistics."""
    log = progress or (lambda _message: None)
    started = time.perf_counter()
    stats = SyncStats()

    vault = Path(settings.vault_path)
    if not vault.is_dir():
        raise FileNotFoundError(f"vault path is not a directory: {vault}")

    log(f"scanning {vault} ...")
    chunks = collect_chunks(
        vault,
        max_chars=settings.chunk_max_chars,
        overlap_chars=settings.chunk_overlap_chars,
        stats=stats,
    )
    stats.chunks_total = len(chunks)

    store.ensure_schema(conn, dim=embedder.dim)  # type: ignore[arg-type]
    known = {} if full else store.existing_hashes(conn)  # type: ignore[arg-type]

    to_embed: list[Chunk] = []
    for chunk in chunks:
        previous = known.get(chunk.id)
        if previous is None:
            stats.chunks_new += 1
            to_embed.append(chunk)
        elif previous != chunk.content_hash:
            stats.chunks_changed += 1
            to_embed.append(chunk)
        else:
            stats.chunks_unchanged += 1

    try:
        if to_embed:
            log(f"embedding {len(to_embed)} chunks with dim={embedder.dim} ...")
            fresh_ids = {chunk.id for chunk in chunks}
            stale_ids = [cid for cid in known if cid not in fresh_ids]
            if stale_ids:
                stats.chunks_deleted = store.delete_chunks(conn, stale_ids)  # type: ignore[arg-type]

            batch_size = max(1, settings.embed_batch_size)
            for start in range(0, len(to_embed), batch_size):
                batch = to_embed[start : start + batch_size]
                vectors = embedder.embed([chunk.embed_text for chunk in batch])
                store.upsert_chunks(conn, batch, vectors)  # type: ignore[arg-type]
                stats.embeddings_computed += len(batch)
                log(f"  {stats.embeddings_computed}/{len(to_embed)} embedded")
        else:
            fresh_ids = {chunk.id for chunk in chunks}
            stale_ids = [cid for cid in known if cid not in fresh_ids]
            if stale_ids:
                stats.chunks_deleted = store.delete_chunks(conn, stale_ids)  # type: ignore[arg-type]
            log("nothing changed; skipping embeddings entirely")
    except Exception:
        conn.rollback()  # type: ignore[attr-defined]
        raise

    stats.elapsed_s = time.perf_counter() - started
    log(stats.summary_line())
    return stats


def chunk_preview(chunks: Sequence[Chunk], limit: int = 5) -> str:
    """Human-readable dump used by ``note-rag ingest --dry-run --preview``."""
    lines = []
    for chunk in chunks[:limit]:
        breadcrumb = " > ".join(chunk.heading_path) or "(root)"
        preview = chunk.text[:120].replace("\n", " ")
        lines.append(f"[{chunk.id}] {chunk.note_path} :: {breadcrumb}\n    {preview}...")
    return "\n".join(lines)
