from __future__ import annotations

import re

import pytesseract
from pdf2image import convert_from_path

from src.canonical.model import BBox, CanonicalDocument, CanonicalItem, ItemType
from src.ingest.base import FormatAdapter, PIDRef
from src.ingest.pdf_native import _classify

DEFAULT_DPI = 200
MIN_OCR_CONFIDENCE = 40.0  # tesseract's own 0-100 scale; below this is mostly
                           # line-art / noise picked up as spurious "words"
JUNK_TOKEN_RE = re.compile(r"^[^A-Za-z0-9]{1,4}$")  # bare punctuation/underscore fragments (e.g. "_", "|_|", ".,")


class ScannedPDFAdapter(FormatAdapter):
    name = "pdf_scanned"

    def __init__(self, dpi: int = DEFAULT_DPI):
        self.dpi = dpi

    def sniff(self, ref: PIDRef) -> bool:
        if not ref.path.lower().endswith((".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            return False
        # Scanned PDFs are the fallback for anything the native adapter
        # rejects (i.e. too little extractable text). Explicit image files
        # are always scanned by definition.
        if not ref.path.lower().endswith(".pdf"):
            return True
        import pdfplumber
        try:
            with pdfplumber.open(ref.path) as pdf:
                if not pdf.pages:
                    return False
                total_chars = sum(len(p.extract_text() or "") for p in pdf.pages)
                return (total_chars / max(len(pdf.pages), 1)) < 40
        except Exception:
            return False

    def ingest(self, ref: PIDRef) -> CanonicalDocument:
        scale = 72.0 / self.dpi  # convert OCR pixel coords back to PDF points
        pages = convert_from_path(ref.path, dpi=self.dpi)

        items: list[CanonicalItem] = []
        page_sizes = {}
        for page_idx, img in enumerate(pages):
            page_sizes[page_idx] = (img.width * scale, img.height * scale)
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            n = len(data["text"])
            for i in range(n):
                text = data["text"][i].strip()
                if not text:
                    continue
                conf_raw = data["conf"][i]
                try:
                    conf_pct = float(conf_raw)
                except (TypeError, ValueError):
                    conf_pct = -1.0
                if conf_pct < MIN_OCR_CONFIDENCE:
                    continue  # drop low-confidence OCR noise before it ever
                                # becomes a CanonicalItem -- see module docstring
                if JUNK_TOKEN_RE.match(text):
                    continue
                conf = max(0.0, conf_pct) / 100.0
                left, top, w, h = (
                    data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                )
                bbox = BBox(
                    x0=left * scale,
                    top=top * scale,
                    x1=(left + w) * scale,
                    bottom=(top + h) * scale,
                )
                items.append(
                    CanonicalItem(
                        item_id=f"{ref.pid}:p{page_idx}:o{i}",
                        page=page_idx,
                        bbox=bbox,
                        item_type=_classify(re.sub(r"[^\w:=.\-]", "", text) or text),
                        text=text,
                        extraction_confidence=conf,
                        source_adapter=self.name,
                        extra={"ocr_engine": "tesseract"},
                    )
                )
        return CanonicalDocument(
            pid=ref.pid,
            source_format=self.name,
            revision_label=ref.revision_label,
            page_count=len(page_sizes),
            page_sizes=page_sizes,
            items=items,
            doc_meta={"path": ref.path, "ocr_dpi": self.dpi},
        )
