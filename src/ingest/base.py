from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from src.canonical.model import CanonicalDocument


@dataclass
class PIDRef:
    """A PID is a resolvable reference to one document revision's bytes +
    metadata. In this reference implementation a PID resolves to a local
    file path, but the interface is deliberately thin so it could just as
    easily resolve to an S3 key, a document-management-system id, etc."""
    pid: str
    path: str
    revision_label: str = ""

    def read_bytes(self) -> bytes:
        return Path(self.path).read_bytes()


class UnsupportedFormatError(Exception):
    pass


class FormatAdapter(ABC):
    """One adapter per source format. `sniff` decides whether this adapter
    can handle a given PIDRef; `ingest` does the actual normalization into
    a CanonicalDocument."""

    name: str = "base"

    @abstractmethod
    def sniff(self, ref: PIDRef) -> bool:
        """Return True if this adapter can handle the given file."""
        raise NotImplementedError

    @abstractmethod
    def ingest(self, ref: PIDRef) -> CanonicalDocument:
        """Normalize the referenced document into a CanonicalDocument."""
        raise NotImplementedError


def detect_format(ref: PIDRef, adapters: list[FormatAdapter]) -> FormatAdapter:
    for adapter in adapters:
        if adapter.sniff(ref):
            return adapter
    raise UnsupportedFormatError(
        f"No registered adapter could handle {ref.path!r}. "
        f"Registered adapters: {[a.name for a in adapters]}"
    )


def resolve_and_ingest(ref: PIDRef, adapters: list[FormatAdapter]) -> CanonicalDocument:
    adapter = detect_format(ref, adapters)
    return adapter.ingest(ref)
