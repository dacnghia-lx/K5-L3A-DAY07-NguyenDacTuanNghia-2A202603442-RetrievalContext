"""Shared data contracts; keep traceability from chunk back to source."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentPage:
    """One traceable source unit ("page").

    The corpus has no physical pages, so a page is one section of a pydoc
    topic, delimited by the reStructuredText headings that Sphinx emitted.
    ``page`` is the 1-based section ordinal inside the topic and
    ``metadata['page_id']`` is the stable ID ``py<version>/<topic>#<page>``.
    """
    document_id: str
    source: str
    page: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    source: str
    page: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
