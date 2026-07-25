from __future__ import annotations

import json
from dataclasses import dataclass

from src.canonical.model import CanonicalDocument
from src.delta.engine import ChangeType, DeltaEntry, split_by_confidence


@dataclass
class DeltaReport:
    pid_a: str
    pid_b: str
    entries: list[DeltaEntry]              # high-confidence, presented as findings
    needs_review: list[DeltaEntry] = None  # low-confidence, flagged separately

    def __post_init__(self):
        if self.needs_review is None:
            self.needs_review = []

    def summary_counts(self) -> dict:
        counts = {"added": 0, "removed": 0, "modified": 0}
        by_type: dict[str, int] = {}
        for e in self.entries:
            counts[e.change_type.value] += 1
            by_type[e.item_type] = by_type.get(e.item_type, 0) + 1
        return {
            "by_change_type": counts,
            "by_item_type": by_type,
            "total": len(self.entries),
            "needs_review": len(self.needs_review),
        }

    def to_json(self) -> dict:
        return {
            "pid_a": self.pid_a,
            "pid_b": self.pid_b,
            "summary": self.summary_counts(),
            "entries": [e.to_dict() for e in self.entries],
            "needs_review": [e.to_dict() for e in self.needs_review],
        }

    def to_markdown(self) -> str:
        summary = self.summary_counts()
        lines = [
            f"# Delta Report: {self.pid_a} -> {self.pid_b}",
            "",
            "## Summary",
            "",
            f"- Total changes: **{summary['total']}**",
            f"- Added: {summary['by_change_type']['added']}",
            f"- Removed: {summary['by_change_type']['removed']}",
            f"- Modified: {summary['by_change_type']['modified']}",
            "",
            "| item type | count |",
            "|---|---|",
        ]
        for item_type, count in sorted(summary["by_item_type"].items()):
            lines.append(f"| {item_type} | {count} |")
        lines.append("")

        for change_type in (ChangeType.MODIFIED, ChangeType.ADDED, ChangeType.REMOVED):
            group = [e for e in self.entries if e.change_type is change_type]
            if not group:
                continue
            lines.append(f"## {change_type.value.capitalize()} ({len(group)})")
            lines.append("")
            by_page: dict[int, list[DeltaEntry]] = {}
            for e in group:
                by_page.setdefault(e.page, []).append(e)
            for page in sorted(by_page):
                lines.append(f"### Page {page + 1}")
                lines.append("")
                for e in sorted(by_page[page], key=lambda x: (x.bbox.top, x.bbox.x0)):
                    loc = f"x={e.bbox.x0:.0f},y={e.bbox.top:.0f}"
                    lines.append(
                        f"- `[{e.entry_id}]` ({e.item_type}, confidence {e.confidence:.2f}, {loc}) "
                        f"{e.description}"
                    )
                lines.append("")

        if self.needs_review:
            lines.append(f"## Needs review -- low confidence ({len(self.needs_review)})")
            lines.append("")
            lines.append(
                "These were detected as candidate changes but fell below the "
                "confidence threshold (see `delta/engine.py:REVIEW_CONFIDENCE_THRESHOLD`), "
                "most often because they come from a low-confidence OCR word on the scanned "
                "side. Listed for human review rather than reported as findings."
            )
            lines.append("")
            for e in sorted(self.needs_review, key=lambda x: (x.page, x.bbox.top, x.bbox.x0))[:30]:
                lines.append(
                    f"- `[{e.entry_id}]` page {e.page + 1}, {e.item_type}, "
                    f"confidence {e.confidence:.2f}: {e.description}"
                )
            if len(self.needs_review) > 30:
                lines.append(f"- ... and {len(self.needs_review) - 30} more (see delta_report.json)")
            lines.append("")
        return "\n".join(lines)

    def chunks_for_retrieval(self) -> list[dict]:
        """One retrievable chunk per delta entry, so chat citations can
        point at a specific delta_report:<entry_id> chunk."""
        out = []
        for e in self.entries + self.needs_review:
            out.append(
                {
                    "chunk_id": f"delta_report:{e.entry_id}",
                    "source": "delta_report",
                    "page": e.page,
                    "text": (
                        f"[{e.change_type.value.upper()}] page {e.page + 1}, {e.item_type}: "
                        f"{e.description} (confidence {e.confidence:.2f})"
                    ),
                }
            )
        return out

    def write(self, out_dir: str) -> tuple[str, str]:
        import os
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, "delta_report.json")
        md_path = os.path.join(out_dir, "delta_report.md")
        with open(json_path, "w") as f:
            json.dump(self.to_json(), f, indent=2, default=str)
        with open(md_path, "w") as f:
            f.write(self.to_markdown())
        return json_path, md_path


def build_report(doc_a: CanonicalDocument, doc_b: CanonicalDocument, entries: list[DeltaEntry]) -> DeltaReport:
    high, low = split_by_confidence(entries)
    return DeltaReport(pid_a=doc_a.pid, pid_b=doc_b.pid, entries=high, needs_review=low)
