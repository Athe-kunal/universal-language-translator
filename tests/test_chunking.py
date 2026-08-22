"""Tests for data_gen.chunking, per the chunking spec's test list."""

import pytest

from data_gen.chunking import (
    _assert_invariants,
    build_units,
    chunk_document,
    reconstruct,
)

ROUND_TRIP_DOCS = [
    "# Heading\n\nA short paragraph.\n",
    (
        "# Title\n\n"
        "Intro paragraph with some words.\n\n"
        "A list follows:\n\n"
        "- item one\n"
        "- item two\n\n"
        "```python\ncode = 1\n```\n\n"
        "> a quoted remark\n"
    ),
    "The cost is $50 per sample and the model uses $x_i \\in \\mathbb{R}$ as input.\n",
    "| a | b |\n| - | - |\n| 1 | 2 |\n",
    "---\n\nAfter a rule.\n",
]


@pytest.mark.parametrize("text", ROUND_TRIP_DOCS)
def test_round_trip_identity(text):
    doc = chunk_document(text, doc_id="doc", source="openthoughts")
    assert reconstruct(doc, {}) == text


@pytest.mark.parametrize("text", ROUND_TRIP_DOCS)
def test_placeholder_round_trip(text):
    doc = chunk_document(text, doc_id="doc", source="openthoughts")
    for unit in doc.units:
        restored = unit.text_protected
        for span in unit.spans:
            restored = restored.replace(span.placeholder, span.original, 1)
        assert restored == unit.text_raw


def test_currency_is_not_math():
    text = "The cost is $50 per sample.\n"
    doc = chunk_document(text, doc_id="doc", source="openthoughts")
    (unit,) = doc.units
    assert not any(span.kind in ("math_inline", "math_block", "amsmath") for span in unit.spans)


def test_dollar_math_is_protected():
    text = "The model uses $x_i \\in \\mathbb{R}$ as input.\n"
    doc = chunk_document(text, doc_id="doc", source="openthoughts")
    (unit,) = doc.units
    math_spans = [span for span in unit.spans if span.kind == "math_inline"]
    assert len(math_spans) == 1
    assert math_spans[0].original == "$x_i \\in \\mathbb{R}$"


def test_fenced_code_is_not_protected_or_translated():
    text = "```python\ncost = $50  # ** not bold **\n\nblank line above\n```\n"
    doc = chunk_document(text, doc_id="doc", source="openthoughts")
    (unit,) = doc.units
    assert unit.kind == "code"
    assert unit.translate is False
    assert unit.spans == []
    assert unit.text_protected == unit.text_raw


def test_nested_list_is_one_unit_when_small():
    text = (
        "- top one\n"
        "  - nested a\n"
        "    - deep i\n"
        "  - nested b\n"
        "- top two\n"
    )
    doc = chunk_document(text, doc_id="doc", source="openthoughts")
    assert len(doc.units) == 1
    assert doc.units[0].kind == "list"
    assert doc.units[0].fingerprint.max_nest_depth >= 2


def test_oversized_nested_list_splits_only_at_top_level():
    items = []
    for i in range(20):
        items.append(f"- top item {i} with some padding words to add length")
        items.append(f"  - nested child {i}a with its own padding words here")
        items.append(f"  - nested child {i}b with its own padding words here")
    text = "\n".join(items) + "\n"

    units_before = build_units(text, doc_id="doc", source="openthoughts")
    (whole,) = units_before
    original_bullet_count = whole.fingerprint.bullet_count

    doc = chunk_document(text, doc_id="doc", source="openthoughts", min_tokens=1, max_tokens=60)
    assert len(doc.units) > 1
    assert all(unit.kind == "list" for unit in doc.units)
    # Splitting at list_item boundaries preserves every bullet (parent + nested).
    assert sum(unit.fingerprint.bullet_count for unit in doc.units) == original_bullet_count
    assert reconstruct(doc, {}) == text


def test_oversized_table_is_not_split():
    header = "| " + " | ".join(f"col{i}" for i in range(8)) + " |\n"
    sep = "| " + " | ".join("-" for _ in range(8)) + " |\n"
    rows = "".join(
        "| " + " | ".join(f"value {r}-{c} padded" for c in range(8)) + " |\n" for r in range(15)
    )
    text = header + sep + rows

    doc = chunk_document(text, doc_id="doc", source="openthoughts", max_tokens=20)
    assert len(doc.units) == 1
    assert doc.units[0].kind == "table"
    assert doc.units[0].token_count > 20


OFFSET_FRAGMENTS = [
    "# H1\n\nPara one.\n\nPara two.\n",
    "- a\n- b\n- c\n",
    "> quoted\n\nafter quote\n",
    "```\nfence\n```\n\nafter fence\n",
    "Para with $x + y$ math and a [link](https://example.com/path).\n",
]


@pytest.mark.parametrize("text", OFFSET_FRAGMENTS)
def test_offset_monotonicity(text):
    doc = chunk_document(text, doc_id="doc", source="openthoughts")
    prev_end = 0
    for unit in doc.units:
        assert unit.char_start >= prev_end
        assert unit.char_end >= unit.char_start
        assert text[unit.char_start : unit.char_end] == unit.text_raw
        prev_end = unit.char_end
    # chunk_document already asserts this internally; re-check explicitly too.
    _assert_invariants(text, doc)


def test_every_placeholder_has_exactly_one_span():
    text = "Cost is $50 and math $x_i \\in \\mathbb{R}$ with `code` and 3.14 percent.\n"
    doc = chunk_document(text, doc_id="doc", source="openthoughts")
    for unit in doc.units:
        ids = [span.placeholder for span in unit.spans]
        assert len(ids) == len(set(ids))
        assert sorted(ids) == unit.fingerprint.placeholder_ids


def test_reconstruct_uses_translations_and_restores_placeholders():
    text = "The cost is $50 today.\n"
    doc = chunk_document(text, doc_id="doc", source="openthoughts")
    (unit,) = doc.units
    (span,) = unit.spans
    assert span.kind == "number"
    translated = unit.text_protected.replace("The cost is", "लागत है")
    out = reconstruct(doc, {unit.unit_id: translated})
    assert span.original in out
    assert unit.unit_id


def test_reconstruct_rejects_dropped_placeholder():
    text = "The cost is $50 today.\n"
    doc = chunk_document(text, doc_id="doc", source="openthoughts")
    (unit,) = doc.units
    bad_translation = unit.text_protected.replace(unit.spans[0].placeholder, "")
    with pytest.raises(AssertionError):
        reconstruct(doc, {unit.unit_id: bad_translation})


def test_discourse_markers_flagged_for_openthoughts():
    text = "Wait, let me reconsider this whole approach before continuing further.\n"
    doc = chunk_document(text, doc_id="doc", source="openthoughts")
    assert doc.units[0].has_discourse_markers is True


def test_discourse_markers_not_flagged_for_naturalreasoning():
    text = "Wait, let me reconsider this whole approach before continuing further.\n"
    doc = chunk_document(text, doc_id="doc", source="naturalreasoning")
    assert doc.units[0].has_discourse_markers is False
