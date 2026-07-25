from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.canonical.model import CanonicalDocument
from src.delta.report import DeltaReport

CHUNK_CHAR_BUDGET = 220


@dataclass
class Chunk:
    chunk_id: str
    source: str      # "pid_a" | "pid_b" | "delta_report"
    page: int
    text: str


def _document_chunks(doc: CanonicalDocument, source_label: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in sorted(doc.page_sizes):
        items = sorted(doc.items_on_page(page), key=lambda it: (it.bbox.top, it.bbox.x0))
        buf: list[str] = []
        buf_len = 0
        idx = 0

        def flush():
            nonlocal buf, buf_len, idx
            if buf:
                text = " ".join(buf)
                chunks.append(Chunk(chunk_id=f"{source_label}:p{page}:c{idx}", source=source_label, page=page, text=text))
                idx += 1
            buf = []
            buf_len = 0

        for it in items:
            piece = it.text.strip()
            if not piece:
                continue
            if buf_len + len(piece) + 1 > CHUNK_CHAR_BUDGET:
                flush()
            buf.append(piece)
            buf_len += len(piece) + 1
        flush()
    return chunks


def _delta_chunks(report: DeltaReport) -> list[Chunk]:
    return [
        Chunk(chunk_id=c["chunk_id"], source=c["source"], page=c["page"], text=c["text"])
        for c in report.chunks_for_retrieval()
    ]


class RetrievalIndex:
    def __init__(self, doc_a: CanonicalDocument, doc_b: CanonicalDocument, report: DeltaReport):
        self.chunks: list[Chunk] = (
            _document_chunks(doc_a, "pid_a")
            + _document_chunks(doc_b, "pid_b")
            + _delta_chunks(report)
        )
        texts = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), token_pattern=r"(?u)\b[\w\-\./#=:%]+\b"
        )
        self.matrix = self.vectorizer.fit_transform(texts) if texts else None

    def search(self, query: str, top_k: int = 6) -> list[tuple[Chunk, float]]:
        if self.matrix is None or not self.chunks:
            return []
        qvec = self.vectorizer.transform([query])
        sims = cosine_similarity(qvec, self.matrix)[0]
        ranked = sorted(range(len(self.chunks)), key=lambda i: sims[i], reverse=True)
        return [(self.chunks[i], float(sims[i])) for i in ranked[:top_k] if sims[i] > 0]
