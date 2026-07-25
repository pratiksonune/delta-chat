from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import math

from src.canonical.model import CanonicalDocument, CanonicalItem

POSITION_WEIGHT = 0.45
TEXT_WEIGHT = 0.55

MATCH_THRESHOLD = 0.5
EXACT_MATCH_TEXT_SIM = 0.90


@dataclass
class Alignment:
    a: CanonicalItem | None  # None => this is an addition (only in B)
    b: CanonicalItem | None  # None => this is a removal (only in A)
    position_sim: float
    text_sim: float

    @property
    def score(self) -> float:
        return POSITION_WEIGHT * self.position_sim + TEXT_WEIGHT * self.text_sim

    @property
    def unchanged(self) -> bool:
        return self.a is not None and self.b is not None and self.text_sim >= EXACT_MATCH_TEXT_SIM


def _text_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def _position_sim(item_a: CanonicalItem, item_b: CanonicalItem, page_diag: float) -> float:
    ax, ay = item_a.bbox.center()
    bx, by = item_b.bbox.center()
    dist = math.hypot(ax - bx, ay - by)
    return max(0.0, 1.0 - dist / page_diag) if page_diag else 0.0


def align_page(
    items_a: list[CanonicalItem],
    items_b: list[CanonicalItem],
    page_size: tuple[float, float],
) -> list[Alignment]:

    page_diag = math.hypot(*page_size) if page_size else 1.0

    candidates: list[tuple[float, int, int]] = []  # (score, idx_a, idx_b)
    for ia, ia_item in enumerate(items_a):
        for ib, ib_item in enumerate(items_b):
            pos_sim = _position_sim(ia_item, ib_item, page_diag)
            # cheap prune: if items are far apart AND text is very different,
            # skip computing the expensive text similarity
            if pos_sim < 0.5:
                text_sim = _text_sim(ia_item.text, ib_item.text)
                if text_sim < 0.5:
                    continue
            else:
                text_sim = _text_sim(ia_item.text, ib_item.text)
            score = POSITION_WEIGHT * pos_sim + TEXT_WEIGHT * text_sim
            if score >= MATCH_THRESHOLD:
                candidates.append((score, ia, ib))

    candidates.sort(key=lambda t: t[0], reverse=True)

    matched_a: set[int] = set()
    matched_b: set[int] = set()
    alignments: list[Alignment] = []

    for score, ia, ib in candidates:
        if ia in matched_a or ib in matched_b:
            continue
        matched_a.add(ia)
        matched_b.add(ib)
        item_a, item_b = items_a[ia], items_b[ib]
        pos_sim = _position_sim(item_a, item_b, page_diag)
        text_sim = _text_sim(item_a.text, item_b.text)
        alignments.append(Alignment(a=item_a, b=item_b, position_sim=pos_sim, text_sim=text_sim))

    for ia, item_a in enumerate(items_a):
        if ia not in matched_a:
            alignments.append(Alignment(a=item_a, b=None, position_sim=0.0, text_sim=0.0))
    for ib, item_b in enumerate(items_b):
        if ib not in matched_b:
            alignments.append(Alignment(a=None, b=item_b, position_sim=0.0, text_sim=0.0))

    return alignments


def align_documents(doc_a: CanonicalDocument, doc_b: CanonicalDocument) -> list[Alignment]:
    all_pages = sorted(set(doc_a.page_sizes) | set(doc_b.page_sizes))
    alignments: list[Alignment] = []
    for page in all_pages:
        items_a = doc_a.items_on_page(page)
        items_b = doc_b.items_on_page(page)
        page_size = doc_a.page_sizes.get(page) or doc_b.page_sizes.get(page) or (1.0, 1.0)
        alignments.extend(align_page(items_a, items_b, page_size))
    return alignments
