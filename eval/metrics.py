from __future__ import annotations

import math
from dataclasses import dataclass

from src.delta.engine import ChangeType, DeltaEntry


@dataclass
class DeltaMatchResult:
    matched_gt_ids: list[str]
    missed_gt_ids: list[str]
    matched_detection_ids: list[str]
    precision: float
    recall: float
    f1: float
    total_detected: int
    total_ground_truth: int


def _center(loc: dict) -> tuple[float, float]:
    return ((loc["x0"] + loc["x1"]) / 2.0, (loc["top"] + loc["bottom"]) / 2.0)


def _entry_center(e: DeltaEntry) -> tuple[float, float]:
    return e.bbox.center()


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def score_delta(ground_truth: list[dict], detected: list[DeltaEntry]) -> DeltaMatchResult:
    matched_gt_ids: list[str] = []
    missed_gt_ids: list[str] = []
    matched_detection_ids: set[str] = set()

    for gt in ground_truth:
        gt_center = _center(gt["location"])
        tol = gt.get("location_tolerance_pt", 40)
        found = False

        if gt["change_type"] == "modified":
            # direct "modified" match
            for e in detected:
                if (
                    e.page == gt["page"]
                    and e.change_type is ChangeType.MODIFIED
                    and _dist(_entry_center(e), gt_center) <= tol
                ):
                    matched_gt_ids.append(gt["gt_id"])
                    matched_detection_ids.add(e.entry_id)
                    found = True
                    break
            if not found:
                # fallback: independent removed + added within tolerance counts
                # as one detected substitution -- see module docstring
                removed = [
                    e for e in detected
                    if e.page == gt["page"] and e.change_type is ChangeType.REMOVED
                    and _dist(_entry_center(e), gt_center) <= tol
                ]
                added = [
                    e for e in detected
                    if e.page == gt["page"] and e.change_type is ChangeType.ADDED
                    and _dist(_entry_center(e), gt_center) <= tol
                ]
                if removed and added:
                    matched_gt_ids.append(gt["gt_id"])
                    matched_detection_ids.add(removed[0].entry_id)
                    matched_detection_ids.add(added[0].entry_id)
                    found = True
        else:
            want_type = ChangeType(gt["change_type"])
            for e in detected:
                if (
                    e.page == gt["page"]
                    and e.change_type is want_type
                    and _dist(_entry_center(e), gt_center) <= tol
                ):
                    matched_gt_ids.append(gt["gt_id"])
                    matched_detection_ids.add(e.entry_id)
                    found = True
                    break

        if not found:
            missed_gt_ids.append(gt["gt_id"])

    total_detected = len(detected)
    total_gt = len(ground_truth)
    tp = len(matched_gt_ids)
    precision = len(matched_detection_ids) / total_detected if total_detected else 0.0
    recall = tp / total_gt if total_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return DeltaMatchResult(
        matched_gt_ids=matched_gt_ids,
        missed_gt_ids=missed_gt_ids,
        matched_detection_ids=sorted(matched_detection_ids),
        precision=precision,
        recall=recall,
        f1=f1,
        total_detected=total_detected,
        total_ground_truth=total_gt,
    )


@dataclass
class ChatScoreResult:
    qa_id: str
    question: str
    answer: str
    keyword_recall: float
    citation_source_hit: bool
    groundedness: float


def score_chat_answer(qa: dict, answer_text: str, citations: list[str], groundedness: float) -> ChatScoreResult:
    keywords = qa.get("expected_keywords", [])
    lower_answer = answer_text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower_answer)
    keyword_recall = hits / len(keywords) if keywords else 1.0

    expected_source = qa.get("expected_citation_source")
    citation_hit = any(c.startswith(expected_source) for c in citations) if expected_source else True

    return ChatScoreResult(
        qa_id=qa["qa_id"],
        question=qa["question"],
        answer=answer_text,
        keyword_recall=keyword_recall,
        citation_source_hit=citation_hit,
        groundedness=groundedness,
    )
