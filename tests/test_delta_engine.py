import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.canonical.model import BBox, CanonicalDocument, CanonicalItem, ItemType
from src.delta.engine import ChangeType, compute_delta, merge_adjacent_word_deltas, split_by_confidence


def make_item(item_id, text, x0, top, item_type=ItemType.TEXT, conf=1.0, page=0):
    return CanonicalItem(
        item_id=item_id,
        page=page,
        bbox=BBox(x0=x0, top=top, x1=x0 + len(text) * 5, bottom=top + 6),
        item_type=item_type,
        text=text,
        extraction_confidence=conf,
        source_adapter="test",
    )


def make_doc(pid, items):
    return CanonicalDocument(
        pid=pid,
        source_format="test",
        revision_label="test",
        page_count=1,
        page_sizes={0: (1000, 1000)},
        items=items,
    )


def test_unchanged_items_produce_no_delta():
    doc_a = make_doc("A", [make_item("a1", "HELLO", 100, 100)])
    doc_b = make_doc("B", [make_item("b1", "HELLO", 100, 100)])
    entries = compute_delta(doc_a, doc_b)
    assert entries == []


def test_modified_dimension_detected():
    doc_a = make_doc("A", [make_item("a1", "257", 851, 60, ItemType.DIMENSION)])
    doc_b = make_doc("B", [make_item("b1", "262", 851, 60, ItemType.DIMENSION)])
    entries = compute_delta(doc_a, doc_b)
    assert len(entries) == 1
    e = entries[0]
    assert e.change_type is ChangeType.MODIFIED
    assert e.before_text == "257"
    assert e.after_text == "262"
    assert e.item_type == "dimension"


def test_removed_item_detected_when_no_counterpart():
    doc_a = make_doc("A", [make_item("a1", "26GT9281", 892, 434, ItemType.TAG)])
    doc_b = make_doc("B", [])
    entries = compute_delta(doc_a, doc_b)
    assert len(entries) == 1
    assert entries[0].change_type is ChangeType.REMOVED
    assert entries[0].before_text == "26GT9281"
    assert entries[0].after_text is None


def test_added_item_detected_when_no_counterpart():
    doc_a = make_doc("A", [])
    doc_b = make_doc("B", [make_item("b1", "NEWNOTE", 955, 728, ItemType.NOTE)])
    entries = compute_delta(doc_a, doc_b)
    assert len(entries) == 1
    assert entries[0].change_type is ChangeType.ADDED
    assert entries[0].after_text == "NEWNOTE"


def test_far_apart_dissimilar_items_are_independent_add_and_remove():
    # unrelated content on opposite corners of the page should NOT be
    # matched to each other just because both changed
    doc_a = make_doc("A", [make_item("a1", "FLARE", 10, 10)])
    doc_b = make_doc("B", [make_item("b1", "PUMP", 900, 900)])
    entries = compute_delta(doc_a, doc_b)
    change_types = sorted(e.change_type.value for e in entries)
    assert change_types == ["added", "removed"]


def test_ocr_style_punctuation_noise_treated_as_unchanged():
    # near-identical text (a stray trailing comma) should not be reported
    # as a false "modified" entry -- see align.py EXACT_MATCH_TEXT_SIM
    doc_a = make_doc("A", [make_item("a1", "VENDOR", 200, 200)])
    doc_b = make_doc("B", [make_item("b1", "VENDOR,", 200, 200, conf=0.7)])
    entries = compute_delta(doc_a, doc_b)
    assert entries == []


def test_split_by_confidence_buckets_low_confidence_separately():
    # similar-enough text at the same position to be matched as "modified"
    # (not independent add/remove), but with low extraction confidence on
    # the B side, so the combined delta confidence should land in "low"
    doc_a = make_doc("A", [make_item("a1", "AAAB", 10, 10)])
    doc_b = make_doc("B", [make_item("b1", "AAAC", 10, 10, conf=0.2)])
    entries = compute_delta(doc_a, doc_b)
    assert len(entries) == 1
    assert entries[0].change_type is ChangeType.MODIFIED
    high, low = split_by_confidence(entries)
    assert len(low) == 1
    assert len(high) == 0


def test_merge_adjacent_word_deltas_combines_same_line_additions():
    doc_a = make_doc("A", [])
    doc_b = make_doc(
        "B",
        [
            make_item("b1", "NEW", 100, 100, ItemType.NOTE),
            make_item("b2", "NOTE", 116, 100, ItemType.NOTE),
            make_item("b3", "HERE", 137, 100, ItemType.NOTE),
        ],
    )
    entries = compute_delta(doc_a, doc_b)
    merged = merge_adjacent_word_deltas(entries)
    assert len(merged) == 1
    assert merged[0].after_text == "NEW NOTE HERE"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
