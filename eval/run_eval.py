from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import score_chat_answer, score_delta
from run import run_pipeline, load_state
from src.chat.answer import answer_question
from src.chat.index import RetrievalIndex
from src.chat.llm import get_provider

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")


def run_eval_for_pair(pair_dir: str, provider_name: str | None = None) -> dict:
    pair_id = os.path.basename(pair_dir)
    gt_path = os.path.join(pair_dir, "ground_truth_delta.json")
    qa_path = os.path.join(pair_dir, "qa.json")

    with open(gt_path) as f:
        gt_data = json.load(f)
    with open(qa_path) as f:
        qa_data = json.load(f)

    out_dir = os.path.join("out", "eval", pair_id)
    result = run_pipeline(gt_data["pid_a"], gt_data["pid_b"], out_dir)
    state = load_state(result["state_path"])
    report = state["report"]

    delta_score = score_delta(gt_data["changes"], report.entries)

    index = RetrievalIndex(state["doc_a"], state["doc_b"], report)
    provider = get_provider(provider_name)
    chat_scores = []
    for qa in qa_data["qa_pairs"]:
        ga = answer_question(qa["question"], index, provider)
        chat_scores.append(score_chat_answer(qa, ga.answer_text, ga.citations, ga.groundedness_score))

    return {
        "pair_id": pair_id,
        "provider": provider.name,
        "model": provider.model,
        "delta": {
            "precision": round(delta_score.precision, 3),
            "recall": round(delta_score.recall, 3),
            "f1": round(delta_score.f1, 3),
            "total_detected": delta_score.total_detected,
            "total_ground_truth": delta_score.total_ground_truth,
            "matched_gt_ids": delta_score.matched_gt_ids,
            "missed_gt_ids": delta_score.missed_gt_ids,
        },
        "chat": {
            "mean_keyword_recall": round(sum(c.keyword_recall for c in chat_scores) / len(chat_scores), 3) if chat_scores else 0.0,
            "citation_source_accuracy": round(sum(1 for c in chat_scores if c.citation_source_hit) / len(chat_scores), 3) if chat_scores else 0.0,
            "mean_groundedness": round(sum(c.groundedness for c in chat_scores) / len(chat_scores), 3) if chat_scores else 0.0,
            "per_question": [
                {
                    "qa_id": c.qa_id,
                    "question": c.question,
                    "keyword_recall": round(c.keyword_recall, 3),
                    "citation_source_hit": c.citation_source_hit,
                    "groundedness": round(c.groundedness, 3),
                    "answer": c.answer,
                }
                for c in chat_scores
            ],
        },
        "report_summary": report.summary_counts(),
    }


def print_scorecard(results: list[dict]):
    print("\n" + "=" * 72)
    print("EVAL SCORECARD")
    print("=" * 72)
    for r in results:
        print(f"\nPair: {r['pair_id']}   (llm: {r['provider']}/{r['model']})")
        print("-" * 72)
        d = r["delta"]
        print(f"  Delta   precision={d['precision']:.3f}  recall={d['recall']:.3f}  f1={d['f1']:.3f}"
              f"   (detected={d['total_detected']}, ground_truth={d['total_ground_truth']})")
        if d["missed_gt_ids"]:
            print(f"          MISSED ground truth: {d['missed_gt_ids']}")
        c = r["chat"]
        print(f"  Chat    mean_keyword_recall={c['mean_keyword_recall']:.3f}"
              f"  citation_source_accuracy={c['citation_source_accuracy']:.3f}"
              f"  mean_groundedness={c['mean_groundedness']:.3f}")
        print("\n  Failure table (candid, not hidden):")
        print(f"    - Delta precision is computed against the FULL high-confidence detected set "
              f"({d['total_detected']} entries) vs. {d['total_ground_truth']} labeled ground-truth "
              f"edits. On this scanned-OCR sample the scanned adapter still emits meaningful noise "
              f"(see src/ingest/pdf_scanned.py docstring); precision reported here is intentionally "
              f"unforgiving, not a bug in the scorer.")
        for c_row in c["per_question"]:
            if c_row["keyword_recall"] < 1.0 or not c_row["citation_source_hit"]:
                print(f"    - {c_row['qa_id']}: keyword_recall={c_row['keyword_recall']:.2f} "
                      f"citation_source_hit={c_row['citation_source_hit']}  Q: {c_row['question']}")
    print("\n" + "=" * 72 + "\n")


def main():
    provider_name = os.environ.get("LLM_PROVIDER")  # default resolved inside get_provider() -> mock
    results = []
    for entry in sorted(os.listdir(DATASETS_DIR)):
        pair_dir = os.path.join(DATASETS_DIR, entry)
        if os.path.isdir(pair_dir):
            results.append(run_eval_for_pair(pair_dir, provider_name))

    print_scorecard(results)

    os.makedirs("out/eval", exist_ok=True)
    scorecard_path = "out/eval/eval_scorecard.json"
    with open(scorecard_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Wrote {scorecard_path}")


if __name__ == "__main__":
    main()
