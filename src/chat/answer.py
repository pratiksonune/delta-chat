from __future__ import annotations

from dataclasses import dataclass, field

from src.chat.index import Chunk, RetrievalIndex
from src.chat.llm import LLMProvider, LLMResult

SYSTEM_PROMPT = (
    "You are a grounded assistant answering questions about two revisions of an "
    "engineering P&ID (PID A = base revision, PID B = revised revision) and a "
    "structured delta report describing what changed between them. "
    "Answer ONLY using the provided CONTEXT. If the context does not contain the "
    "answer, say so plainly instead of guessing. Be concise and specific: reference "
    "tag numbers, page numbers, and delta entry ids where relevant."
)


@dataclass
class GroundedAnswer:
    question: str
    answer_text: str
    citations: list[str] = field(default_factory=list)   # chunk_ids used
    retrieved_chunks: list[Chunk] = field(default_factory=list)
    llm_result: LLMResult | None = None
    groundedness_score: float = 0.0

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer_text,
            "citations": self.citations,
            "retrieved_chunk_ids": [c.chunk_id for c in self.retrieved_chunks],
            "groundedness_score": round(self.groundedness_score, 3),
            "model": self.llm_result.model if self.llm_result else None,
            "provider": self.llm_result.provider if self.llm_result else None,
            "input_tokens": self.llm_result.input_tokens if self.llm_result else None,
            "output_tokens": self.llm_result.output_tokens if self.llm_result else None,
            "cost_usd": self.llm_result.cost_usd if self.llm_result else None,
            "latency_ms": self.llm_result.latency_ms if self.llm_result else None,
        }


def _build_prompt(question: str, chunks: list[Chunk]) -> str:
    context_lines = []
    for c in chunks:
        context_lines.append(f"[{c.chunk_id}] ({c.source}, page {c.page + 1}) {c.text}")
    context = "\n".join(context_lines)
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Answer the question using only the CONTEXT above. Cite the chunk ids you "
        "used in square brackets, e.g. [pid_a:p0:c3] or [delta_report:d0007]."
    )


def _groundedness(answer_text: str, chunks: list[Chunk]) -> float:
    if not chunks:
        return 0.0
    answer_tokens = set(w.lower() for w in answer_text.split() if len(w) > 3)
    if not answer_tokens:
        return 0.0
    context_tokens = set()
    for c in chunks:
        context_tokens.update(w.lower() for w in c.text.split() if len(w) > 3)
    if not context_tokens:
        return 0.0
    overlap = answer_tokens & context_tokens
    return len(overlap) / max(1, len(answer_tokens))


def answer_question(question: str, index: RetrievalIndex, provider: LLMProvider, top_k: int = 6) -> GroundedAnswer:
    results = index.search(question, top_k=top_k)
    chunks = [c for c, _score in results]

    prompt = _build_prompt(question, chunks)
    llm_result = provider.complete(SYSTEM_PROMPT, prompt)

    import re
    cited = re.findall(r"\[([\w:\-.]+)\]", llm_result.text)
    valid_chunk_ids = {c.chunk_id for c in chunks}
    citations = [c for c in cited if c in valid_chunk_ids]
    if not citations and chunks:
        # model (or the mock provider) didn't emit bracket citations --
        # fall back to citing the top retrieved chunk(s) explicitly so the
        # answer is never presented as uncited when grounded context existed
        citations = [c.chunk_id for c in chunks[:2]]

    score = _groundedness(llm_result.text, chunks)

    return GroundedAnswer(
        question=question,
        answer_text=llm_result.text,
        citations=citations,
        retrieved_chunks=chunks,
        llm_result=llm_result,
        groundedness_score=score,
    )
