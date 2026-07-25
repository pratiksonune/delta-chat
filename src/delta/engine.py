from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.canonical.model import BBox, CanonicalDocument
from src.delta.align import Alignment, align_documents


class ChangeType(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


@dataclass
class DeltaEntry:
    entry_id: str
    change_type: ChangeType
    item_type: str
    page: int
    bbox: BBox
    description: str
    confidence: float
    before_text: str | None = None
    after_text: str | None = None

    def to_dict(self):
        return {
            "entry_id": self.entry_id,
            "change_type": self.change_type.value,
            "item_type": self.item_type,
            "page": self.page,
            "location": self.bbox.to_dict(),
            "description": self.description,
            "confidence": round(self.confidence, 3),
            "before_text": self.before_text,
            "after_text": self.after_text,
        }


def _describe(change_type: ChangeType, item_type: str, before: str | None, after: str | None) -> str:
    if change_type is ChangeType.ADDED:
        return f'New {item_type} added: "{after}"'
    if change_type is ChangeType.REMOVED:
        return f'{item_type.capitalize()} removed: "{before}"'
    return f'{item_type.capitalize()} changed from "{before}" to "{after}"'


def compute_delta(doc_a: CanonicalDocument, doc_b: CanonicalDocument) -> list[DeltaEntry]:
    alignments = align_documents(doc_a, doc_b)
    entries: list[DeltaEntry] = []
    counter = 0

    for al in alignments:
        counter += 1
        if al.a is not None and al.b is not None:
            if al.unchanged:
                continue  # not a delta -- content is the same on both sides
            item_type = al.b.item_type.value
            conf = al.score * min(al.a.extraction_confidence, al.b.extraction_confidence)
            entries.append(
                DeltaEntry(
                    entry_id=f"d{counter:04d}",
                    change_type=ChangeType.MODIFIED,
                    item_type=item_type,
                    page=al.b.page,
                    bbox=al.b.bbox,
                    description=_describe(ChangeType.MODIFIED, item_type, al.a.text, al.b.text),
                    confidence=conf,
                    before_text=al.a.text,
                    after_text=al.b.text,
                )
            )
        elif al.a is not None and al.b is None:
            entries.append(
                DeltaEntry(
                    entry_id=f"d{counter:04d}",
                    change_type=ChangeType.REMOVED,
                    item_type=al.a.item_type.value,
                    page=al.a.page,
                    bbox=al.a.bbox,
                    description=_describe(ChangeType.REMOVED, al.a.item_type.value, al.a.text, None),
                    confidence=al.a.extraction_confidence,
                    before_text=al.a.text,
                    after_text=None,
                )
            )
        elif al.a is None and al.b is not None:
            entries.append(
                DeltaEntry(
                    entry_id=f"d{counter:04d}",
                    change_type=ChangeType.ADDED,
                    item_type=al.b.item_type.value,
                    page=al.b.page,
                    bbox=al.b.bbox,
                    description=_describe(ChangeType.ADDED, al.b.item_type.value, None, al.b.text),
                    confidence=al.b.extraction_confidence,
                    before_text=None,
                    after_text=al.b.text,
                )
            )

    entries.sort(key=lambda e: (e.page, e.bbox.top, e.bbox.x0))
    for i, e in enumerate(entries, start=1):
        e.entry_id = f"d{i:04d}"
    return entries


# Below this, a "modified" entry is more likely to be extraction noise
# (an OCR misread paired up with its nearest native-text neighbor) than a
# real content change. Entries below this line are still computed and
# reported -- see report.py -- just bucketed as "needs review" instead of
# presented as confident findings. This threshold is a judgment call, not a
# derived constant; documented in the README and revisited against the
# eval set's labeled ground truth.
REVIEW_CONFIDENCE_THRESHOLD = 0.55


def split_by_confidence(entries: list[DeltaEntry]) -> tuple[list[DeltaEntry], list[DeltaEntry]]:
    """Returns (high_confidence, needs_review)."""
    high = [e for e in entries if e.confidence >= REVIEW_CONFIDENCE_THRESHOLD]
    low = [e for e in entries if e.confidence < REVIEW_CONFIDENCE_THRESHOLD]
    return high, low


def merge_adjacent_word_deltas(entries: list[DeltaEntry], x_gap: float = 6.0, y_tol: float = 2.0) -> list[DeltaEntry]:
    """Optional post-pass: merges consecutive same-type, same-change-type
    word-level deltas on the same line into a single phrase-level delta, so
    e.g. a 3-word note addition is reported as one entry instead of three.
    Kept separate from compute_delta so callers can opt in/out and so the
    raw word-level deltas stay available for the eval harness (ground truth
    is labeled at word granularity for pair_001)."""
    if not entries:
        return entries
    merged: list[DeltaEntry] = []
    used = [False] * len(entries)
    by_page: dict[int, list[int]] = {}
    for i, e in enumerate(entries):
        by_page.setdefault(e.page, []).append(i)

    for page, idxs in by_page.items():
        idxs.sort(key=lambda i: (entries[i].bbox.top, entries[i].bbox.x0))
        for i in idxs:
            if used[i]:
                continue
            group = [i]
            used[i] = True
            cur = entries[i]
            for j in idxs:
                if used[j]:
                    continue
                cand = entries[j]
                if (
                    cand.change_type == cur.change_type
                    and cand.item_type == cur.item_type
                    and abs(cand.bbox.top - cur.bbox.top) <= y_tol
                    and 0 <= (cand.bbox.x0 - cur.bbox.x1) <= x_gap
                ):
                    group.append(j)
                    used[j] = True
                    cur = cand
            if len(group) == 1:
                merged.append(entries[i])
            else:
                items = [entries[k] for k in group]
                x0 = min(it.bbox.x0 for it in items)
                x1 = max(it.bbox.x1 for it in items)
                top = min(it.bbox.top for it in items)
                bottom = max(it.bbox.bottom for it in items)
                before = " ".join(it.before_text for it in items if it.before_text) or None
                after = " ".join(it.after_text for it in items if it.after_text) or None
                merged.append(
                    DeltaEntry(
                        entry_id=items[0].entry_id,
                        change_type=items[0].change_type,
                        item_type=items[0].item_type,
                        page=page,
                        bbox=BBox(x0, top, x1, bottom),
                        description=_describe(items[0].change_type, items[0].item_type, before, after),
                        confidence=sum(it.confidence for it in items) / len(items),
                        before_text=before,
                        after_text=after,
                    )
                )
    merged.sort(key=lambda e: (e.page, e.bbox.top, e.bbox.x0))
    for i, e in enumerate(merged, start=1):
        e.entry_id = f"d{i:04d}"
    return merged
