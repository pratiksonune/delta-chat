from __future__ import annotations

from src.canonical.model import CanonicalDocument
from src.ingest.base import FormatAdapter, PIDRef


class DWGAdapter(FormatAdapter):
    name = "dwg"

    def sniff(self, ref: PIDRef) -> bool:
        return ref.path.lower().endswith((".dwg", ".dxf"))

    def ingest(self, ref: PIDRef) -> CanonicalDocument:
        raise NotImplementedError(
            "DWG ingestion is stubbed behind the FormatAdapter interface but not "
            "implemented in this submission. A real implementation would shell out "
            "to a DWG->DXF converter and parse the result with ezdxf -- see the "
            "module docstring for the intended design. The adapter is registered "
            "and will be selected (sniff() returns True) for .dwg/.dxf inputs so "
            "the failure is explicit rather than falling through to the wrong "
            "adapter."
        )
