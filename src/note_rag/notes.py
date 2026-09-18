"""Obsidian-aware note parsing and heading-aware chunking.

Design notes (keep these when you write the ADR):

* A chunk keeps ``note_path`` + ``heading_path`` as metadata. Without the heading breadcrumb
  an embedding of a bullet list is nearly meaningless.
* ``Chunk.id`` is derived from ``(note_path, heading_path, char_start)`` so it is *stable*
  across content edits that do not move offsets. ``Chunk.content_hash`` is what drives
  incremental re-embedding: same hash => skip the expensive embed call entirely.
* ``Chunk.embed_text`` prepends title + heading breadcrumb to the body. This measurably
  improves recall for short notes; measure it yourself in the eval harness.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#([^\]|]+))?(?:\|([^\]]+))?\]\]")
TAG_RE = re.compile(r"(?:^|\s)#([A-Za-z0-9_\-/\u4e00-\u9fff]{1,64})")
HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n", re.DOTALL)


def _sha1(*parts: object) -> str:
    h = hashlib.sha1(usedforsecurity=False)
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


@dataclass(slots=True, frozen=True)
class Chunk:
    """One embeddable unit of a note."""

    id: str
    note_path: str
    title: str
    heading_path: tuple[str, ...]
    text: str
    char_start: int
    char_end: int
    content_hash: str

    @property
    def embed_text(self) -> str:
        breadcrumb = " > ".join(self.heading_path)
        return f"{self.title}\n{breadcrumb}\n\n{self.text}".strip()


@dataclass(slots=True)
class Note:
    """A parsed markdown note."""

    path: str
    title: str
    body: str
    tags: tuple[str, ...] = ()
    links: tuple[str, ...] = ()
    frontmatter: dict[str, object] = field(default_factory=dict)


def parse_note(rel_path: str, raw: str) -> Note:
    """Parse raw markdown into a :class:`Note`.

    ``rel_path`` must be vault-relative and use forward slashes.
    """
    frontmatter: dict[str, object] = {}
    body = raw
    if match := FRONTMATTER_RE.match(raw):
        try:
            loaded = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            loaded = None
        if isinstance(loaded, dict):
            frontmatter = {str(k): v for k, v in loaded.items()}
        body = raw[match.end() :]

    title = ""
    for line in body.splitlines():
        if (match := HEADING_RE.match(line)) and len(match.group(1)) == 1:
            title = match.group(2).strip()
            break
    if not title:
        fm_title = frontmatter.get("title")
        title = str(fm_title).strip() if fm_title else Path(rel_path).stem

    tags: list[str] = []
    fm_tags = frontmatter.get("tags") or frontmatter.get("tag")
    if isinstance(fm_tags, str):
        tags.extend(t.strip() for t in fm_tags.replace(",", " ").split() if t.strip())
    elif isinstance(fm_tags, list):
        tags.extend(str(t).strip() for t in fm_tags if str(t).strip())
    tags.extend(TAG_RE.findall(body))

    links: list[str] = []
    for target in WIKILINK_RE.findall(body):
        name = target[0].strip()
        if name and name not in links:
            links.append(name)

    return Note(
        path=rel_path,
        title=title,
        body=body,
        tags=tuple(dict.fromkeys(tags)),
        links=tuple(links),
        frontmatter=frontmatter,
    )


def _sections(body: str) -> list[tuple[tuple[str, ...], str, int]]:
    """Split the body into ``(heading_path, text, offset)`` sections.

    A section's text starts right after its heading line so that headings are not
    duplicated in the embedded text (the breadcrumb carries them).
    """
    out: list[tuple[tuple[str, ...], str, int]] = []
    stack: list[str] = []
    current_start = 0
    current_head: tuple[str, ...] = ()

    def flush(end: int) -> None:
        text = body[current_start:end].strip()
        if text:
            out.append((current_head, text, current_start))

    offset = 0
    for line in body.splitlines(keepends=True):
        if match := HEADING_RE.match(line.rstrip("\r\n")):
            flush(offset)
            level = len(match.group(1))
            stack[level - 1 :] = [match.group(2).strip()]
            current_head = tuple(stack)
            current_start = offset + len(line)
            offset += len(line)
            continue
        offset += len(line)

    flush(len(body))
    return out


def _split_long(
    text: str, base_offset: int, max_chars: int, overlap_chars: int
) -> list[tuple[str, int, int]]:
    """Split an oversized section on paragraph boundaries, with character overlap.

    ``overlap_chars`` is the number of characters *repeated* at the start of the next
    piece, so consecutive pieces advance by ``max_chars - overlap_chars``. Offsets are
    tracked arithmetically (no ``str.find``) because repeated paragraphs would otherwise
    resolve to the wrong position and corrupt the chunk ids.
    """
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")
    advance = max_chars - overlap_chars

    pieces: list[tuple[str, int, int]] = []
    buffer = ""
    buffer_start = base_offset
    cursor = 0

    for para in text.split("\n\n"):
        para_start = cursor + base_offset
        cursor += len(para) + 2  # +2 for the "\n\n" separator
        candidate = f"{buffer}\n\n{para}" if buffer else para

        if len(candidate) <= max_chars:
            if not buffer:
                buffer_start = para_start
            buffer = candidate
            continue

        if buffer.strip():
            pieces.append((buffer.strip(), buffer_start, buffer_start + len(buffer)))
        buffer = ""

        # A single paragraph longer than the budget: hard split with overlap.
        piece_start = para_start
        remaining = para
        while len(remaining) > max_chars:
            head = remaining[:max_chars]
            pieces.append((head.strip(), piece_start, piece_start + len(head)))
            remaining = remaining[advance:]
            piece_start += advance
        buffer = remaining
        buffer_start = piece_start

    if buffer.strip():
        pieces.append((buffer.strip(), buffer_start, buffer_start + len(buffer)))
    return pieces


def chunk_note(
    note: Note,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> tuple[Chunk, ...]:
    """Chunk a note into stable, heading-aware, embeddable units."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")

    candidates: list[tuple[tuple[str, ...], str, int, int]] = []
    for heading_path, text, start in _sections(note.body):
        if len(text) <= max_chars:
            candidates.append((heading_path, text, start, start + len(text)))
            continue
        for piece, pstart, pend in _split_long(text, start, max_chars, overlap_chars):
            candidates.append((heading_path, piece, pstart, pend))

    # Greedily merge neighbouring pieces so that tiny sections are not embedded alone.
    #
    # Merging is restricted to pieces under the *same* heading path. Merging across a
    # heading boundary silently destroys the breadcrumb (one chunk cannot honestly claim
    # two different sections), and it also collapses a whole note into a single chunk,
    # which would make every edit invalidate every chunk and kill incremental indexing.
    # Whether merging sibling subsections under a shared *parent* breadcrumb retrieves
    # better is exactly the kind of question the eval harness exists to answer.
    merged: list[tuple[tuple[str, ...], str, int, int]] = []
    for heading_path, text, start, end in candidates:
        if merged:
            prev_head, prev_text, prev_start, _ = merged[-1]
            if prev_head == heading_path and len(prev_text) + len(text) + 2 <= max_chars:
                merged[-1] = (prev_head, f"{prev_text}\n\n{text}", prev_start, end)
                continue
        merged.append((heading_path, text, start, end))

    chunks: list[Chunk] = []
    for heading_path, text, start, end in merged:
        chunk_id = _sha1(note.path, " > ".join(heading_path), start)[:16]
        chunks.append(
            Chunk(
                id=chunk_id,
                note_path=note.path,
                title=note.title,
                heading_path=heading_path,
                text=text,
                char_start=start,
                char_end=end,
                content_hash=_sha1(text)[:16],
            )
        )
    return tuple(chunks)


MARKDOWN_SUFFIXES = {".md", ".markdown"}


def iter_vault_files(vault: Path, *, include_hidden: bool = False) -> list[Path]:
    """List markdown files in a vault, skipping Obsidian internals.

    Only hidden *directories* are skipped (``.obsidian``, ``.trash``, ``.git``, ...). A
    markdown file whose own name starts with a dot is real content and must be indexed.

    This used to test every path part, filename included. That silently dropped 2 of 1947
    notes in ``data/corpus``, where the source directory is flattened into the file name
    (``.agents/notes/AGENTS.md`` becomes ``.agents__notes__AGENTS__<hash>.md`` -- a file,
    not a hidden path). The eval set still labelled them, so 3 questions were unwinnable for
    every mode while nothing reported an error. See docs/reviews/review-001, S2.
    """
    skip_dirs = {".obsidian", ".trash", ".git", "node_modules"}
    files: list[Path] = []
    for path in vault.rglob("*"):
        if path.suffix.lower() not in MARKDOWN_SUFFIXES or not path.is_file():
            continue
        rel = path.relative_to(vault)
        if any(part in skip_dirs for part in rel.parts):
            continue
        # rel.parts[:-1] is the directory chain; rel.parts[-1] is the file name itself.
        if not include_hidden and any(part.startswith(".") for part in rel.parts[:-1]):
            continue
        files.append(path)
    return sorted(files)
