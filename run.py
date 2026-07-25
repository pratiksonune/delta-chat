from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.canonical.model import CanonicalDocument
from src.chat.answer import answer_question
from src.chat.index import RetrievalIndex
from src.chat.llm import get_provider
from src.delta.engine import compute_delta, merge_adjacent_word_deltas
from src.delta.report import DeltaReport, build_report
from src.ingest.base import PIDRef, resolve_and_ingest
from src.ingest.dwg import DWGAdapter
from src.ingest.pdf_native import NativePDFAdapter
from src.ingest.pdf_scanned import ScannedPDFAdapter
from src.markup.overlay import render_markup
from src.observability.logging import get_logger, log_event
from src.observability.tracing import Trace, new_request_id

ADAPTERS = [NativePDFAdapter(), ScannedPDFAdapter(), DWGAdapter()]


def run_pipeline(pid_a_path: str, pid_b_path: str, out_dir: str, request_id: str | None = None) -> dict:
    request_id = request_id or new_request_id()
    logger = get_logger(request_id)
    trace = Trace(request_id=request_id, trace_meta={"pid_a": pid_a_path, "pid_b": pid_b_path})
    os.makedirs(out_dir, exist_ok=True)

    log_event(logger, request_id, "pipeline", "start", pid_a=pid_a_path, pid_b=pid_b_path)

    with trace.span("ingest_pid_a", path=pid_a_path) as span:
        ref_a = PIDRef(pid="PID-A", path=pid_a_path, revision_label="RevA")
        doc_a = resolve_and_ingest(ref_a, ADAPTERS)
        span.metadata.update({"format": doc_a.source_format, "items": len(doc_a.items)})
        log_event(logger, request_id, "ingest", "pid_a_ingested", format=doc_a.source_format, items=len(doc_a.items))

    with trace.span("ingest_pid_b", path=pid_b_path) as span:
        ref_b = PIDRef(pid="PID-B", path=pid_b_path, revision_label="RevB")
        doc_b = resolve_and_ingest(ref_b, ADAPTERS)
        span.metadata.update({"format": doc_b.source_format, "items": len(doc_b.items)})
        log_event(logger, request_id, "ingest", "pid_b_ingested", format=doc_b.source_format, items=len(doc_b.items))

    with trace.span("delta") as span:
        raw_entries = compute_delta(doc_a, doc_b)
        entries = merge_adjacent_word_deltas(raw_entries)
        span.metadata.update({"raw_entries": len(raw_entries), "merged_entries": len(entries)})
        log_event(logger, request_id, "delta", "computed", raw_entries=len(raw_entries), merged_entries=len(entries))

    with trace.span("report") as span:
        report = build_report(doc_a, doc_b, entries)
        json_path, md_path = report.write(out_dir)
        span.metadata.update(report.summary_counts())
        log_event(logger, request_id, "report", "written", json_path=json_path, md_path=md_path, **report.summary_counts())

    state_path = os.path.join(out_dir, "state.pkl")
    with open(state_path, "wb") as f:
        pickle.dump({"doc_a": doc_a, "doc_b": doc_b, "report": report, "pid_b_path": pid_b_path}, f)

    trace_path = trace.flush()
    log_event(logger, request_id, "pipeline", "done", trace_path=str(trace_path))

    return {
        "request_id": request_id,
        "out_dir": out_dir,
        "json_report": json_path,
        "md_report": md_path,
        "state_path": state_path,
        "trace_path": str(trace_path),
        "summary": report.summary_counts(),
    }


def load_state(state_path: str):
    with open(state_path, "rb") as f:
        return pickle.load(f)


def run_chat(state_path: str, question: str | None, provider_name: str | None, request_id: str | None = None):
    request_id = request_id or new_request_id()
    logger = get_logger(request_id)
    trace = Trace(request_id=request_id, trace_meta={"state_path": state_path})

    state = load_state(state_path)
    doc_a: CanonicalDocument = state["doc_a"]
    doc_b: CanonicalDocument = state["doc_b"]
    report: DeltaReport = state["report"]

    with trace.span("build_index") as span:
        index = RetrievalIndex(doc_a, doc_b, report)
        span.metadata.update({"chunks": len(index.chunks)})

    provider = get_provider(provider_name)

    def ask(q: str):
        with trace.span("retrieval", question=q) as span:
            results = index.search(q)
            span.metadata.update({"top_k": len(results)})
        with trace.span("llm_call", model=provider.model, provider=provider.name) as span:
            ga = answer_question(q, index, provider)
            span.metadata.update(
                {
                    "input_tokens": ga.llm_result.input_tokens,
                    "output_tokens": ga.llm_result.output_tokens,
                    "cost_usd": ga.llm_result.cost_usd,
                    "groundedness": ga.groundedness_score,
                }
            )
        log_event(
            logger, request_id, "chat", "answered",
            question=q, citations=ga.citations, cost_usd=ga.llm_result.cost_usd,
            groundedness=ga.groundedness_score,
        )
        return ga

    if question:
        ga = ask(question)
        print(f"\nQ: {question}")
        print(f"A: {ga.answer_text}")
        print(f"Citations: {ga.citations}")
        print(f"Groundedness: {ga.groundedness_score:.2f}  Cost: ${ga.llm_result.cost_usd:.5f}  "
              f"Latency: {ga.llm_result.latency_ms:.1f}ms  Provider/model: {ga.llm_result.provider}/{ga.llm_result.model}")
        trace.flush()
        return ga

    print(f"Grounded chat over {doc_a.pid} <-> {doc_b.pid}. Provider={provider.name}/{provider.model}. Ctrl-D to quit.")
    try:
        while True:
            q = input("\n> ").strip()
            if not q:
                continue
            ga = ask(q)
            print(ga.answer_text)
            print(f"[citations: {ga.citations}  groundedness: {ga.groundedness_score:.2f}]")
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        trace.flush()


def run_markup(state_path: str, out_dir: str):
    state = load_state(state_path)
    report: DeltaReport = state["report"]
    pid_b_path = state["pid_b_path"]
    os.makedirs(out_dir, exist_ok=True)
    paths = render_markup(pid_b_path, report.entries, out_dir)
    print("Wrote markup:")
    for p in paths:
        print(f"  {p}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="delta-chat: document delta + grounded chat")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pipe = sub.add_parser("pipeline", help="ingest both PIDs, compute delta, write report")
    p_pipe.add_argument("--pid-a", required=True)
    p_pipe.add_argument("--pid-b", required=True)
    p_pipe.add_argument("--out", required=True)

    p_chat = sub.add_parser("chat", help="grounded chat over a computed delta")
    p_chat.add_argument("--state", required=True)
    p_chat.add_argument("--question", default=None, help="single question (non-interactive)")
    p_chat.add_argument("--provider", default=None, help="mock | hf_local | anthropic")

    p_markup = sub.add_parser("markup", help="render delta markup overlay onto PID B")
    p_markup.add_argument("--state", required=True)
    p_markup.add_argument("--out", required=True)

    args = parser.parse_args()

    if args.command == "pipeline":
        result = run_pipeline(args.pid_a, args.pid_b, args.out)
        print(json.dumps(result, indent=2))
    elif args.command == "chat":
        run_chat(args.state, args.question, args.provider)
    elif args.command == "markup":
        run_markup(args.state, args.out)


if __name__ == "__main__":
    main()
