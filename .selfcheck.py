"""Offline self-check (no pytest, no Postgres, no network).

Run:  python .selfcheck.py

Kept alongside the pytest suite on purpose: this one runs on a bare interpreter with no
installed dependencies, so it still works when `uv sync` has not happened yet. It proves the
two invariants that are easy to break silently -- chunk-id stability and incremental
re-embedding -- and needs nothing but stdlib + PyYAML.
"""

from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# psycopg is not installed in the system interpreter; provide a stub so the store module imports.
if "psycopg" not in sys.modules:
    stub = types.ModuleType("psycopg")
    stub.Connection = object  # type: ignore[attr-defined]
    stub.connect = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.modules["psycopg"] = stub

from note_rag import ingest
from note_rag.config import Settings
from note_rag.notes import chunk_note, iter_vault_files, parse_note

SAMPLE = """---
title: 检索笔记
tags: [rag, retrieval]
---

# 检索笔记

## 混合检索

向量检索和 BM25 各有所长，用 RRF 融合排名。
参见 [[向量索引]] 和 [[评测方法#指标]]。

## 评测

recall@5 是最常用的指标。
"""

failures: list[str] = []


def check(label: str, condition: bool) -> None:
    print(("  PASS  " if condition else "  FAIL  ") + label)
    if not condition:
        failures.append(label)


print("[notes]")
note = parse_note("10-项目/检索.md", SAMPLE)
check("frontmatter title parsed", note.title == "检索笔记")
check("frontmatter tags parsed", "rag" in note.tags and "retrieval" in note.tags)
check("wikilinks extracted", "向量索引" in note.links and "评测方法" in note.links)
check("frontmatter stripped from body", "---" not in note.body)

chunks = chunk_note(note, max_chars=400, overlap_chars=50)
paths = [c.heading_path for c in chunks]
check("heading breadcrumbs", ("检索笔记", "混合检索") in paths and ("检索笔记", "评测") in paths)
check("heading line not embedded", all(not c.text.startswith("#") for c in chunks))
check("embed_text has breadcrumb", ">" in chunks[0].embed_text)

again = [c.id for c in chunk_note(note, max_chars=400, overlap_chars=50)]
check("chunk ids stable", [c.id for c in chunks] == again)

edited = parse_note(
    "10-项目/检索.md", SAMPLE.replace("recall@5 是最常用的指标。", "MRR 也很重要。")
)
old = {c.id: c.content_hash for c in chunks}
new = {c.id: c.content_hash for c in chunk_note(edited, max_chars=400, overlap_chars=50)}
shared = [cid for cid in old if cid in new]
check("edit detected on changed chunk", any(old[c] != new[c] for c in shared))
check("unrelated chunks keep their hash", any(old[c] == new[c] for c in shared))

long_body = "# T\n\n## S\n\n" + "\n\n".join(f"段落 {i} " + "x" * 200 for i in range(20))
long_chunks = chunk_note(parse_note("n.md", long_body), max_chars=500, overlap_chars=80)
check(
    "long section split within budget",
    len(long_chunks) > 1 and all(len(c.text) <= 500 for c in long_chunks),
)

tiny = chunk_note(
    parse_note("n.md", "# T\n\n## A\n\n短\n\n## B\n\n也短\n"), max_chars=400, overlap_chars=50
)
check(
    "no merge across heading boundary",
    len(tiny) == 2 and {c.heading_path[-1] for c in tiny} == {"A", "B"},
)
same_head = chunk_note(
    parse_note("n.md", "# T\n\n## S\n\n短\n\n还是同一节\n"), max_chars=400, overlap_chars=50
)
check("same-heading pieces merged", len(same_head) == 1)

print("\n[ingest incremental]")
vault = Path(__file__).parent / ".selfcheck_vault"
if vault.exists():
    import shutil

    shutil.rmtree(vault)
vault.mkdir()
(vault / "a.md").write_text("# A\n\n## A1\n\nalpha 内容\n", encoding="utf-8")
(vault / "b.md").write_text("# B\n\nbeta 内容\n", encoding="utf-8")
check("vault markdown discovered", len(iter_vault_files(vault)) == 2)

table: dict[str, str] = {}
ingest.store.ensure_schema = lambda conn, dim=384: None  # type: ignore[assignment]
ingest.store.existing_hashes = lambda conn: dict(table)  # type: ignore[assignment]


def _upsert(conn, chunks_, embeddings, *, embedder_name=""):
    for chunk in chunks_:
        table[chunk.id] = chunk.content_hash
    return len(chunks_)


def _delete(conn, ids):
    return sum(1 for cid in list(ids) if table.pop(cid, None) is not None)


ingest.store.upsert_chunks = _upsert  # type: ignore[assignment]
ingest.store.delete_chunks = _delete  # type: ignore[assignment]


class Conn:
    def rollback(self) -> None:
        pass


class Embedder:
    dim = 8

    def __init__(self) -> None:
        self.calls = 0
        self.texts = 0

    def embed(self, texts):
        self.calls += 1
        self.texts += len(texts)
        return [[0.1] * self.dim for _ in texts]


settings = Settings(vault_path=vault, embed_dim=8, chunk_max_chars=400, chunk_overlap_chars=50)
first = ingest.sync_vault(settings=settings, conn=Conn(), embedder=Embedder())
check("first sync embeds everything", first.embeddings_computed == first.chunks_total > 0)

second_embedder = Embedder()
second = ingest.sync_vault(settings=settings, conn=Conn(), embedder=second_embedder)
check(
    "unchanged re-sync computes 0 embeddings",
    second.embeddings_computed == 0 and second_embedder.calls == 0,
)

(vault / "a.md").write_text("# A\n\n## A1\n\nalpha 内容已修改\n", encoding="utf-8")
third_embedder = Embedder()
third = ingest.sync_vault(settings=settings, conn=Conn(), embedder=third_embedder)
check("edit re-embeds only the touched chunk", 0 < third.embeddings_computed < first.chunks_total)
check("unchanged chunks accounted for", third.chunks_unchanged > 0)

(vault / "b.md").unlink()
fourth = ingest.sync_vault(settings=settings, conn=Conn(), embedder=Embedder())
check("deleted note removes its chunks", fourth.chunks_deleted > 0)
check("no stale chunks left", all("b.md" not in cid for cid in table))

shutil.rmtree(vault, ignore_errors=True)

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    raise SystemExit(1)
print("all offline self-checks passed")
