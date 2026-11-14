"""Fixed-size and structure-aware chunking. Chunks never cross a page boundary,
so every chunk maps back to exactly one (document_id, page)."""
from __future__ import annotations

import hashlib
import re

from .models import Chunk, DocumentPage

_BLANK = re.compile(r"\n\s*\n")


def _make_chunk(page: DocumentPage, strategy_key: str, index: int, text: str,
                start: int, end: int, **extra) -> Chunk:
    page_id = page.metadata["page_id"]
    meta = dict(page.metadata)
    meta.update({
        "chunk_index": index,
        "char_start": start,
        "char_end": end,
        "chunk_chars": len(text),
        "chunk_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
    })
    meta.update(extra)
    return Chunk(
        chunk_id=f"{strategy_key}:{page_id}:c{index:03d}",
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=meta,
    )


def fixed_size(page: DocumentPage, size: int, overlap: int, strategy_key: str = "fixed") -> list[Chunk]:
    """Character windows of ``size`` with ``overlap``; no attention to structure."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    text = page.text
    step = size - overlap
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(_make_chunk(page, strategy_key, len(chunks), piece, start, end))
        if end == len(text):
            break
        start += step
    return chunks


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for m in _BLANK.finditer(text):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return spans


def _split_oversized(text: str, max_size: int) -> list[tuple[int, int]]:
    """Split one long paragraph at line breaks; hard-split only a line > max_size.

    Returns (start, end) offsets relative to ``text``.
    """
    out: list[tuple[int, int]] = []
    cur_s, pos = 0, 0
    for line in text.split("\n"):
        line_end = pos + len(line)
        if line_end - cur_s > max_size and pos > cur_s:
            out.append((cur_s, pos - 1))          # close before this line
            cur_s = pos
        while line_end - cur_s > max_size:          # single over-long line
            out.append((cur_s, cur_s + max_size))
            cur_s += max_size
        pos = line_end + 1
    if len(text) > cur_s:
        out.append((cur_s, len(text)))
    return out


def by_paragraph(page: DocumentPage, max_size: int, strategy_key: str = "para",
                 add_heading: bool = True) -> list[Chunk]:
    """Group consecutive paragraphs (blank-line separated) up to ``max_size``.

    - A paragraph never gets cut unless it alone exceeds ``max_size``; then it
      is split at line breaks (code/grammar blocks), and only a single line
      longer than ``max_size`` is hard-split.
    - With ``add_heading`` the chunk text starts with a context line
      ``[<topic title> > <section>]`` when the chunk does not already begin
      with the section heading. This is part of strategy B's chunking.
    """
    text = page.text
    units: list[tuple[int, int]] = []
    for s, e in _paragraph_spans(text):
        if e - s > max_size:
            sub = _split_oversized(text[s:e], max_size)
            units.extend((s + a, s + b) for a, b in sub)
        else:
            units.append((s, e))

    groups: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = None
    for s, e in units:
        if cur is None:
            cur = (s, e)
        elif e - cur[0] <= max_size:
            cur = (cur[0], e)
        else:
            groups.append(cur)
            cur = (s, e)
    if cur is not None:
        groups.append(cur)

    topic_title = page.metadata.get("topic_title", "")
    section = page.metadata.get("section", "")
    context = f"[{topic_title} > {section}]" if section and section != topic_title else f"[{topic_title}]"
    chunks: list[Chunk] = []
    for s, e in groups:
        body = text[s:e].strip()
        if not body:
            continue
        starts_with_heading = s == 0
        chunk_text = body if (starts_with_heading or not add_heading) else f"{context}\n{body}"
        chunks.append(_make_chunk(page, strategy_key, len(chunks), chunk_text, s, e,
                                  heading_prefixed=not starts_with_heading and add_heading))
    return chunks
