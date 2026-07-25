from __future__ import annotations
import re
import pdfplumber

from src.canonical.model import BBox, CanonicalDocument, CanonicalItem, ItemType
from src.ingest.base import FormatAdapter, PIDRef

TAG_RE = re.compile(r"^\d{2}-?[A-Z]{2,4}-?\d{3,5}[A-Z]?$")
DIM_RE = re.compile(r"^(H|HH|L|LL|SP|SD)?[:=]?\s*-?\d+(\.\d+)?$", re.IGNORECASE)
NOTE_IDX_RE = re.compile(r"^\d{1,2}\.$")

MIN_CHARS_PER_PAGE = 40


def _classify(token: str) -> ItemType:
    if NOTE_IDX_RE.match(token):
        return ItemType.NOTE
    if TAG_RE.match(token):
        return ItemType.TAG
    if DIM_RE.match(token) or re.search(r"\d", token) and any(
        u in token.upper() for u in ("BAR", "BARG", "°C", "MM", "KW", "KG")
    ):
        return ItemType.DIMENSION
    return ItemType.TEXT


class NativePDFAdapter(FormatAdapter):
    name = "pdf_native"

    def sniff(self, ref: PIDRef) -> bool:
        if not ref.path.lower().endswith(".pdf"):
            return False
        try:
            with pdfplumber.open(ref.path) as pdf:
                if not pdf.pages:
                    return False
                total_chars = sum(len(p.extract_text() or "") for p in pdf.pages)
                return (total_chars / max(len(pdf.pages), 1)) >= MIN_CHARS_PER_PAGE
        except Exception:
            return False

    def ingest(self, ref: PIDRef) -> CanonicalDocument:
        items: list[CanonicalItem] = []
        page_sizes = {}
        with pdfplumber.open(ref.path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_sizes[page_idx] = (page.width, page.height)
                words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
                for i, w in enumerate(words):
                    item_type = _classify(w["text"])
                    items.append(
                        CanonicalItem(
                            item_id=f"{ref.pid}:p{page_idx}:w{i}",
                            page=page_idx,
                            bbox=BBox(w["x0"], w["top"], w["x1"], w["bottom"]),
                            item_type=item_type,
                            text=w["text"],
                            extraction_confidence=1.0,
                            source_adapter=self.name,
                        )
                    )
        return CanonicalDocument(
            pid=ref.pid,
            source_format=self.name,
            revision_label=ref.revision_label,
            page_count=len(page_sizes),
            page_sizes=page_sizes,
            items=items,
            doc_meta={"path": ref.path},
        )
