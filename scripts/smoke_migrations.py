#!/usr/bin/env python
"""CI smoke test: the schema must be creatable against a real Postgres.

Why this exists as a script instead of an inline ``python -c``:

* the previous inline one-liner called ``ensure_schema()`` without a connection and broke CI
  the first time it ran -- an argument that only appears when you actually execute it;
* unit tests never touch Postgres, so nothing else in the suite can catch a schema or
  connection-parameter regression.

It also asserts the empty-corpus contract, so a change that silently makes ``corpus_stats``
report something else on a fresh database fails loudly here.

Usage:
    NOTE_RAG_DATABASE_URL=postgresql://... uv run python scripts/smoke_migrations.py
"""

from __future__ import annotations

import os
import sys

from note_rag import store

EMBED_DIM = 384


def main() -> int:
    url = os.environ.get("NOTE_RAG_DATABASE_URL")
    if not url:
        print("NOTE_RAG_DATABASE_URL is not set", file=sys.stderr)
        return 1

    conn = store.connect(url)
    try:
        store.ensure_schema(conn, dim=EMBED_DIM)
        print(f"schema ok (vector dim = {EMBED_DIM})")

        stats = store.corpus_stats(conn)
        print(f"corpus stats on a fresh database: {stats}")
        if stats != {"chunks": 0, "notes": 0}:
            print(f"unexpected stats on a fresh database: {stats}", file=sys.stderr)
            return 1

        # Idempotency: running the migration twice must not raise.
        store.ensure_schema(conn, dim=EMBED_DIM)
        print("ensure_schema is idempotent")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
