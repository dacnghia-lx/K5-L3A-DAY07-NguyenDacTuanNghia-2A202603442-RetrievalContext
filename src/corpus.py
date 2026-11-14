"""Day 6 connection: same corpus, normalization, chunking (Day 6 strategy B/C:
paragraph chunks <=1000 chars + heading context line), same embedding model
(WordLlama l2_supercat, 256 d) and same Chroma collection settings."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .day6.chunking import by_paragraph
from .day6.ingestion import flatten, load_pydoc_topics
from .day6.models import DocumentPage

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHUNKS_JSON = DATA_DIR / "chunks.json"
BENCHMARK = ROOT / "benchmark" / "questions.json"
RESULTS = ROOT / "results"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "day06_B_para1000"          # identical name/params to Day 6
SOURCES = [("3.11", "data/raw/pydoc_topics_py3.11.py"), ("3.13", "data/raw/pydoc_topics_py3.13.py")]


def load_pages(errors: list | None = None) -> list[DocumentPage]:
    pages: list[DocumentPage] = []
    for version, rel in SOURCES:
        pages.extend(load_pydoc_topics(ROOT / rel, version, rel, errors))
    return pages


def build_chunks(pages: list[DocumentPage]) -> list[dict[str, Any]]:
    """Day 6 strategy B chunks converted to the Day 7 chunk schema."""
    out = []
    for p in pages:
        for c in by_paragraph(p, 1000, "B", True):
            out.append({"id": c.chunk_id, "text": c.text, "source_id": c.document_id, "source": c.source,
                        "page": c.page, "metadata": c.metadata})
    return out


def load_chunks() -> list[dict[str, Any]]:
    if not CHUNKS_JSON.exists():
        raise FileNotFoundError("data/chunks.json missing: run `python -m scripts.build_index`")
    return json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))


def load_benchmark() -> list[dict[str, Any]]:
    return json.loads(BENCHMARK.read_text(encoding="utf-8"))["questions"]


def chunk_has_evidence(chunk: dict, gold: list[dict]) -> bool:
    """Hit definition (same for every method): chunk is on a gold (source_id, page)
    and contains the full evidence string (whitespace-insensitive)."""
    text = flatten(chunk["text"])
    return any(chunk["source_id"] == g["source_id"] and chunk["page"] == g["page"]
               and flatten(g["evidence"]) in text for g in gold)
