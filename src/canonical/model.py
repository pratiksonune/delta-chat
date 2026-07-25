from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import hashlib
import json


class ItemType(str, Enum):
    TEXT = "text"           # a word/phrase/label on the drawing
    NOTE = "note"           # a numbered note in a notes list
    TAG = "tag"             # an instrument/equipment tag (e.g. 26-PIT-9055)
    DIMENSION = "dimension"  # a numeric dimension/setpoint/pressure/temp value
    TABLE_CELL = "table_cell"
    GEOMETRY = "geometry"   # a vector shape / symbol (line, box, line style)


@dataclass(frozen=True)
class BBox:
    """Bounding box in PDF points, top-down origin (y grows downward),
    matching pdfplumber's convention. x0/x1 are left/right, top/bottom are
    the vertical extent measured from the top of the page."""
    x0: float
    top: float
    x1: float
    bottom: float

    def as_tuple(self):
        return (self.x0, self.top, self.x1, self.bottom)

    def center(self):
        return ((self.x0 + self.x1) / 2.0, (self.top + self.bottom) / 2.0)

    def to_dict(self):
        return {"x0": self.x0, "top": self.top, "x1": self.x1, "bottom": self.bottom}


@dataclass
class CanonicalItem:
    """One atomic, located piece of content on a page/sheet."""
    item_id: str
    page: int
    bbox: BBox
    item_type: ItemType
    text: str
    extraction_confidence: float = 1.0  # 1.0 = deterministic extraction (native text)
    source_adapter: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "item_id": self.item_id,
            "page": self.page,
            "bbox": self.bbox.to_dict(),
            "item_type": self.item_type.value,
            "text": self.text,
            "extraction_confidence": self.extraction_confidence,
            "source_adapter": self.source_adapter,
            "extra": self.extra,
        }


@dataclass
class CanonicalDocument:
    """A whole document/drawing revision, normalized."""
    pid: str
    source_format: str          # "pdf_native" | "pdf_scanned" | "dwg"
    revision_label: str
    page_count: int
    page_sizes: dict            # {page_index: (width, height)}
    items: list[CanonicalItem]
    doc_meta: dict = field(default_factory=dict)

    def content_hash(self) -> str:
        """A stable hash of the normalized content, independent of item
        ordering. Used so the eval harness / observability can quickly tell
        whether two ingests of the "same" document actually agree."""
        payload = sorted(
            (it.page, round(it.bbox.x0, 1), round(it.bbox.top, 1), it.item_type.value, it.text)
            for it in self.items
        )
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_dict(self):
        return {
            "pid": self.pid,
            "source_format": self.source_format,
            "revision_label": self.revision_label,
            "page_count": self.page_count,
            "page_sizes": self.page_sizes,
            "content_hash": self.content_hash(),
            "items": [it.to_dict() for it in self.items],
            "doc_meta": self.doc_meta,
        }

    def items_on_page(self, page: int):
        return [it for it in self.items if it.page == page]
