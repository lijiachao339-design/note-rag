from __future__ import annotations

from pathlib import Path

import pytest

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


def test_parse_frontmatter_tags_title_and_links() -> None:
    note = parse_note("10-项目/检索.md", SAMPLE)
    assert note.title == "检索笔记"
    assert "rag" in note.tags
    assert "retrieval" in note.tags
    assert "向量索引" in note.links
    assert "评测方法" in note.links
    assert note.frontmatter["title"] == "检索笔记"
    assert "---" not in note.body


def test_title_falls_back_to_first_h1_then_to_filename() -> None:
    no_frontmatter = "# 标题来自 H1\n\n正文\n"
    assert parse_note("a/b.md", no_frontmatter).title == "标题来自 H1"
    assert parse_note("a/b.md", "没有标题的正文\n").title == "b"


def test_inline_tags_are_collected() -> None:
    note = parse_note("x.md", "正文 #数据库 #rag/检索\n")
    assert "数据库" in note.tags
    assert "rag/检索" in note.tags


def test_broken_frontmatter_does_not_crash() -> None:
    note = parse_note("x.md", "---\ntags: [oops\n---\n\n# 标题\n")
    assert note.title == "标题"


def test_chunk_heading_path_follows_the_heading_stack() -> None:
    note = parse_note("n.md", SAMPLE)
    chunks = chunk_note(note, max_chars=400, overlap_chars=50)
    paths = [chunk.heading_path for chunk in chunks]
    assert ("检索笔记", "混合检索") in paths
    assert ("检索笔记", "评测") in paths


def test_chunk_text_excludes_the_heading_line() -> None:
    note = parse_note("n.md", SAMPLE)
    chunk = next(
        c for c in chunk_note(note, max_chars=400, overlap_chars=50) if c.heading_path[-1] == "评测"
    )
    assert not chunk.text.startswith("#")
    assert "recall@5" in chunk.text


def test_embed_text_carries_title_and_breadcrumb() -> None:
    note = parse_note("n.md", SAMPLE)
    chunk = chunk_note(note, max_chars=400, overlap_chars=50)[0]
    assert "检索笔记" in chunk.embed_text
    assert ">" in chunk.embed_text


def test_chunk_ids_are_stable_across_rechunking() -> None:
    note = parse_note("n.md", SAMPLE)
    first = [chunk.id for chunk in chunk_note(note, max_chars=400, overlap_chars=50)]
    second = [chunk.id for chunk in chunk_note(note, max_chars=400, overlap_chars=50)]
    assert first == second
    assert len(set(first)) == len(first)


def test_content_hash_changes_only_for_the_edited_chunk() -> None:
    original = parse_note("n.md", SAMPLE)
    edited = parse_note("n.md", SAMPLE.replace("recall@5 是最常用的指标。", "MRR 也很重要。"))
    old = {c.id: c.content_hash for c in chunk_note(original, max_chars=400, overlap_chars=50)}
    new = {c.id: c.content_hash for c in chunk_note(edited, max_chars=400, overlap_chars=50)}
    same_id = [cid for cid in old if cid in new]
    assert same_id
    changed = [cid for cid in same_id if old[cid] != new[cid]]
    unchanged = [cid for cid in same_id if old[cid] == new[cid]]
    assert changed, "the edited chunk must be detected as changed"
    assert unchanged, "unrelated chunks must keep their hash so they are not re-embedded"


def test_long_section_is_split_with_overlap_and_stays_within_budget() -> None:
    body = "# T\n\n## S\n\n" + "\n\n".join(f"paragraph {i} " + "x" * 200 for i in range(20))
    note = parse_note("n.md", body)
    chunks = chunk_note(note, max_chars=500, overlap_chars=80)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 500 for chunk in chunks)
    # Neighbouring pieces of the same heading share no gap in coverage.
    assert all(chunk.heading_path == ("T", "S") for chunk in chunks)


def test_tiny_sections_under_one_heading_are_merged() -> None:
    body = "# T\n\n## S\n\n短\n\n还是同一节\n"
    note = parse_note("n.md", body)
    chunks = chunk_note(note, max_chars=400, overlap_chars=50)
    assert len(chunks) == 1
    assert "短" in chunks[0].text
    assert "还是同一节" in chunks[0].text


def test_merging_never_crosses_a_heading_boundary() -> None:
    """A chunk must not claim two different sections: that would corrupt the breadcrumb
    and collapse the note into one chunk, defeating incremental indexing."""
    body = "# T\n\n## A\n\n短\n\n## B\n\n也短\n"
    note = parse_note("n.md", body)
    chunks = chunk_note(note, max_chars=400, overlap_chars=50)
    assert len(chunks) == 2
    assert {chunk.heading_path[-1] for chunk in chunks} == {"A", "B"}


def test_chunk_offsets_point_back_into_the_body() -> None:
    note = parse_note("n.md", SAMPLE)
    for chunk in chunk_note(note, max_chars=400, overlap_chars=50):
        assert 0 <= chunk.char_start < chunk.char_end <= len(note.body)
        assert chunk.text[:20] in note.body


def test_overlapping_split_advances_and_covers_the_text() -> None:
    paragraph = "".join(f"词{i:03d} " for i in range(200))
    note = parse_note("n.md", f"# T\n\n## S\n\n{paragraph}\n")
    chunks = chunk_note(note, max_chars=300, overlap_chars=60)
    assert len(chunks) > 1
    starts = [chunk.char_start for chunk in chunks]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts), "split must advance, never loop"
    assert all(len(chunk.text) <= 300 for chunk in chunks)


def test_invalid_parameters_are_rejected() -> None:
    note = parse_note("n.md", SAMPLE)
    with pytest.raises(ValueError, match="max_chars must be positive"):
        chunk_note(note, max_chars=0)
    with pytest.raises(ValueError, match="overlap_chars must be smaller"):
        chunk_note(note, max_chars=100, overlap_chars=100)


def test_iter_vault_files_skips_obsidian_internals(tmp_path: Path) -> None:
    (tmp_path / "note.md").write_text("# hi", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.markdown").write_text("# deep", encoding="utf-8")
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "workspace.md").write_text("junk", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    found = {path.relative_to(tmp_path).as_posix() for path in iter_vault_files(tmp_path)}
    assert found == {"note.md", "sub/deep.markdown"}
