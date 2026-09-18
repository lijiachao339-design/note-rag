"""Ingestion tests. No database required: ``store`` is monkeypatched with an in-memory fake.

The headline invariant -- *re-syncing an unchanged vault computes zero embeddings* -- is the
kind of test that turns a feature claim into a defensible resume bullet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from note_rag import ingest, store
from note_rag.config import Settings
from note_rag.notes import Chunk


class FakeConn:
    """Enough of a psycopg connection for the ingest path."""

    def __init__(self) -> None:
        self.rollbacks = 0

    def rollback(self) -> None:
        self.rollbacks += 1


class CountingEmbedder:
    dim = 8

    def __init__(self) -> None:
        self.calls = 0
        self.texts = 0

    def embed(self, texts):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.texts += len(texts)
        return [[0.1] * self.dim for _ in texts]


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    (tmp_path / "a.md").write_text("# A\n\n## A1\n\nalpha 内容\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\n\nbeta 内容\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Replace store functions with an in-memory table."""
    table: dict[str, str] = {}

    monkeypatch.setattr(ingest.store, "ensure_schema", lambda conn, dim=384: None)
    monkeypatch.setattr(ingest.store, "existing_hashes", lambda conn: dict(table))

    def upsert(conn, chunks: list[Chunk], embeddings, *, embedder_name: str = "") -> int:
        for chunk in chunks:
            table[chunk.id] = chunk.content_hash
        return len(chunks)

    def delete(conn, ids) -> int:
        return sum(1 for cid in list(ids) if table.pop(cid, None) is not None)

    monkeypatch.setattr(ingest.store, "upsert_chunks", upsert)
    monkeypatch.setattr(ingest.store, "delete_chunks", delete)
    return table


def _settings(vault: Path) -> Settings:
    return Settings(vault_path=vault, embed_dim=8, chunk_max_chars=400, chunk_overlap_chars=50)


def test_dry_run_collects_chunks_without_touching_the_database(vault: Path) -> None:
    stats = ingest.SyncStats()
    chunks = ingest.collect_chunks(vault, max_chars=400, overlap_chars=50, stats=stats)
    assert stats.files_scanned == 2
    assert stats.notes_failed == 0
    assert chunks
    assert all(isinstance(chunk, Chunk) for chunk in chunks)


def test_unreadable_file_is_reported_not_fatal(tmp_path: Path) -> None:
    # Note the body: a file containing only a heading legitimately yields zero chunks,
    # so the readable file here needs actual content for this test to mean anything.
    (tmp_path / "ok.md").write_text("# ok\n\n这里是有内容的正文。\n", encoding="utf-8")
    (tmp_path / "bad.md").write_bytes(b"\xff\xfe\x00\x00invalid utf-8")
    stats = ingest.SyncStats()
    chunks = ingest.collect_chunks(tmp_path, max_chars=400, overlap_chars=50, stats=stats)
    assert stats.files_scanned == 2
    assert stats.notes_failed == 1
    assert stats.failures, "坏文件必须被记录到 failures 而不是被静默跳过"
    assert "bad.md" in stats.failures[0]
    assert chunks, "可读的那个文件必须仍然产出切块"


def test_heading_only_note_yields_no_chunks(tmp_path: Path) -> None:
    # Documents the behaviour the test above depends on: headings carry metadata, not content.
    (tmp_path / "empty.md").write_text("# 只有标题\n", encoding="utf-8")
    stats = ingest.SyncStats()
    chunks = ingest.collect_chunks(tmp_path, max_chars=400, overlap_chars=50, stats=stats)
    assert chunks == []
    assert stats.files_scanned == 1
    assert stats.notes_failed == 0


def test_first_sync_embeds_everything(vault: Path, fake_store: dict[str, str]) -> None:
    embedder = CountingEmbedder()
    stats = ingest.sync_vault(settings=_settings(vault), conn=FakeConn(), embedder=embedder)
    assert stats.chunks_new == stats.chunks_total > 0
    assert stats.embeddings_computed == stats.chunks_total
    assert len(fake_store) == stats.chunks_total


def test_second_unchanged_sync_computes_zero_embeddings(
    vault: Path, fake_store: dict[str, str]
) -> None:
    settings = _settings(vault)
    ingest.sync_vault(settings=settings, conn=FakeConn(), embedder=CountingEmbedder())

    embedder = CountingEmbedder()
    stats = ingest.sync_vault(settings=settings, conn=FakeConn(), embedder=embedder)

    assert stats.embeddings_computed == 0
    assert embedder.calls == 0
    assert stats.chunks_unchanged == stats.chunks_total
    assert stats.chunks_new == 0
    assert stats.chunks_changed == 0


def test_editing_one_note_re_embeds_only_that_note(vault: Path, fake_store: dict[str, str]) -> None:
    settings = _settings(vault)
    first = ingest.sync_vault(settings=settings, conn=FakeConn(), embedder=CountingEmbedder())

    (vault / "a.md").write_text("# A\n\n## A1\n\nalpha 内容已修改\n", encoding="utf-8")
    embedder = CountingEmbedder()
    second = ingest.sync_vault(settings=settings, conn=FakeConn(), embedder=embedder)

    assert 0 < second.embeddings_computed < first.chunks_total
    assert second.chunks_unchanged > 0
    assert embedder.texts == second.embeddings_computed


def test_deleting_a_note_removes_its_chunks(vault: Path, fake_store: dict[str, str]) -> None:
    settings = _settings(vault)
    ingest.sync_vault(settings=settings, conn=FakeConn(), embedder=CountingEmbedder())
    before = len(fake_store)

    (vault / "b.md").unlink()
    stats = ingest.sync_vault(settings=settings, conn=FakeConn(), embedder=CountingEmbedder())

    assert stats.chunks_deleted > 0
    assert len(fake_store) < before
    assert all("b.md" not in chunk_id for chunk_id in fake_store)


def test_full_reindex_re_embeds_everything(vault: Path, fake_store: dict[str, str]) -> None:
    settings = _settings(vault)
    ingest.sync_vault(settings=settings, conn=FakeConn(), embedder=CountingEmbedder())
    stats = ingest.sync_vault(
        settings=settings, conn=FakeConn(), embedder=CountingEmbedder(), full=True
    )
    assert stats.embeddings_computed == stats.chunks_total


def test_missing_vault_directory_fails_loudly(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "does-not-exist")
    with pytest.raises(FileNotFoundError):
        ingest.sync_vault(settings=settings, conn=FakeConn(), embedder=CountingEmbedder())


def test_stats_serialise_for_logging(vault: Path, fake_store: dict[str, str]) -> None:
    stats = ingest.sync_vault(
        settings=_settings(vault), conn=FakeConn(), embedder=CountingEmbedder()
    )
    payload = stats.as_dict()
    assert payload["chunks_total"] == stats.chunks_total
    assert isinstance(payload["elapsed_s"], float)
    assert "chunks" in stats.summary_line()


def test_upsert_is_called_in_batches(vault: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A large vault must not build one giant statement."""
    calls: list[int] = []
    monkeypatch.setattr(ingest.store, "ensure_schema", lambda conn, dim=384: None)
    monkeypatch.setattr(ingest.store, "existing_hashes", lambda conn: {})
    monkeypatch.setattr(
        ingest.store,
        "upsert_chunks",
        lambda conn, chunks, embeddings, *, embedder_name="": (
            calls.append(len(chunks)) or len(chunks)
        ),
    )
    monkeypatch.setattr(ingest.store, "delete_chunks", lambda conn, ids: 0)

    settings = Settings(vault_path=vault, embed_dim=8, embed_batch_size=1)
    ingest.sync_vault(settings=settings, conn=FakeConn(), embedder=CountingEmbedder())

    assert len(calls) > 1
    assert all(size == 1 for size in calls)


def test_store_symbols_used_by_ingest_exist() -> None:
    """Guards against typing drift between the ingest module and the store layer."""
    for name in ("ensure_schema", "existing_hashes", "upsert_chunks", "delete_chunks"):
        assert callable(getattr(store, name))
