"""Tests for the eval harness helpers that decide what counts as relevant.

These are pure functions over one dataset row, so they need neither a database nor a corpus.
"""

from __future__ import annotations

import pytest

from note_rag.cli import _grades_for


def test_a_list_of_notes_means_every_note_has_grade_one() -> None:
    row = {"id": "q1", "relevant_notes": ["a.md", "b.md"]}

    assert _grades_for(row) == {"a.md": 1.0, "b.md": 1.0}


def test_an_empty_list_is_a_legal_unanswerable_question() -> None:
    """`[]` is deliberate, not missing data: it exposes retrievers that always answer."""
    assert _grades_for({"id": "n1", "relevant_notes": []}) == {}


def test_an_object_carries_graded_relevance() -> None:
    """The source note (2) outranks its translated twin (1); see review-001 S8."""
    row = {"id": "q1", "relevant_notes": {"src.md": 2, "src.zh.md": 1}}

    assert _grades_for(row) == {"src.md": 2.0, "src.zh.md": 1.0}


def test_a_scalar_is_rejected_rather_than_silently_scored_zero() -> None:
    with pytest.raises(ValueError, match="relevant_notes"):
        _grades_for({"id": "q1", "relevant_notes": "a.md"})
