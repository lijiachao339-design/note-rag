"""MCP (Model Context Protocol) server exposing the vault to any MCP-capable agent.

This is the piece that makes the project different from a tutorial RAG: after ``note-rag mcp``
is registered, Claude Code / DeepSeek Harness / any MCP client can search your own notes
mid-task. Tools are intentionally small and composable:

* ``search_notes``  -- semantic + keyword hybrid retrieval, returns note paths and snippets
* ``read_note``     -- read a note (optionally a single heading section) by path
* ``list_recent``   -- recently modified notes, for "what was I working on" questions

Register it with:

    claude mcp add note-rag -- note-rag mcp        # Claude Code
    # then: /mcp  ->  verify the three tools are listed

Design notes for the interview: stdio transport (no port, no auth, process lifetime tied to
the client) is the right default for a local-first tool. Only reach for SSE/HTTP when the
server must be shared or needs to outlive the client.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from note_rag.config import Settings, get_settings
from note_rag.notes import FRONTMATTER_RE
from note_rag.retriever import Retriever

MAX_SNIPPET_CHARS = 600


def _format_hits(hits: list[Any]) -> str:
    if not hits:
        return "No matching notes found."
    blocks = []
    for index, hit in enumerate(hits, start=1):
        breadcrumb = " > ".join(hit.heading_path)
        snippet = hit.text[:MAX_SNIPPET_CHARS].strip()
        blocks.append(
            f"{index}. {hit.note_path}\n"
            f"   title: {hit.title}\n"
            f"   section: {breadcrumb or '(root)'}\n"
            f"   score: {hit.score:.4f}\n"
            f"   {snippet}"
        )
    return "\n\n".join(blocks)


def _safe_note_path(vault: Path, rel_path: str) -> Path:
    """Resolve a note path and refuse anything that escapes the vault.

    Path traversal is the classic bug in "let the model read a file" tools: the model can
    pass ``../../.ssh/id_rsa``. Resolve first, then verify containment.
    """
    candidate = (vault / rel_path).resolve()
    root = vault.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"path escapes the vault: {rel_path!r}")
    if not candidate.is_file():
        raise FileNotFoundError(f"note not found: {rel_path}")
    if candidate.suffix.lower() not in {".md", ".markdown"}:
        raise ValueError(f"not a markdown note: {rel_path}")
    return candidate


def _strip_frontmatter(text: str) -> str:
    match = FRONTMATTER_RE.match(text)
    return text[match.end() :] if match else text


def build_server(settings: Settings | None = None) -> Any:
    """Create the MCP server.

    Imported lazily so the core package has no hard MCP dependency.

    API note: ``mcp`` 2.x renamed ``FastMCP`` to ``MCPServer``
    (``mcp.server.mcpserver``). Code copied from v1 examples fails with
    ``ModuleNotFoundError: No module named 'mcp.server.fastmcp'``.
    """
    from mcp.server.mcpserver import MCPServer

    settings = settings or get_settings()
    retriever = Retriever.from_settings(settings)
    retriever.build_keyword_index()
    vault = Path(settings.vault_path)

    mcp = MCPServer("note-rag")

    @mcp.tool()
    def search_notes(query: str, k: int = 5, mode: str = "hybrid") -> str:
        """Search the Obsidian vault. Prefer short, keyword-rich queries.

        Args:
            query: natural language or keyword query.
            k: number of notes to return (1-20).
            mode: "hybrid" (default), "vector", "keyword" or "hybrid_rerank".
        """
        k = max(1, min(int(k), 20))
        hits = retriever.search_notes(query, k=k, mode=mode)  # type: ignore[arg-type]
        return _format_hits(hits)

    @mcp.tool()
    def read_note(note_path: str, heading: str = "") -> str:
        """Read a note from the vault.

        Args:
            note_path: vault-relative path, exactly as returned by search_notes.
            heading: optional heading text; returns only that section when given.
        """
        path = _safe_note_path(vault, note_path)
        body = _strip_frontmatter(path.read_text(encoding="utf-8"))
        if not heading:
            return body[:20_000]
        wanted = heading.strip().lower()
        lines = body.splitlines()
        collected: list[str] = []
        capturing = False
        for line in lines:
            if line.startswith("#"):
                title = line.lstrip("#").strip().lower()
                if capturing:
                    break
                capturing = title == wanted
                if capturing:
                    collected.append(line)
                continue
            if capturing:
                collected.append(line)
        if not collected:
            raise ValueError(f"heading {heading!r} not found in {note_path}")
        return "\n".join(collected).strip()[:20_000]

    @mcp.tool()
    def list_recent(limit: int = 10) -> str:
        """List the most recently modified notes, newest first."""
        limit = max(1, min(int(limit), 50))
        files = [
            path
            for path in vault.rglob("*.md")
            if path.is_file() and ".obsidian" not in path.parts and ".trash" not in path.parts
        ]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        rows = [
            {
                "note_path": path.relative_to(vault).as_posix(),
                "modified": path.stat().st_mtime,
            }
            for path in files[:limit]
        ]
        return json.dumps(rows, ensure_ascii=False, indent=2)

    return mcp


def main() -> None:
    """Entry point for ``note-rag mcp`` (stdio transport)."""
    build_server().run()


if __name__ == "__main__":
    main()
