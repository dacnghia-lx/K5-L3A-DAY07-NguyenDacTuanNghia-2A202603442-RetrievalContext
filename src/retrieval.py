"""Retrieval methods for Day 7. Every retriever returns chunks in the Day 7
schema plus ``score``, ``rank`` and ``method`` so traces are comparable.

1. DenseRetriever  - Day 6 Chroma collection + WordLlama query embedding (cosine).
2. BM25Retriever   - Okapi BM25 over the same chunks, code-aware tokenizer.
3. HybridRetriever - Reciprocal Rank Fusion of dense and BM25 candidate lists.
Options (improvements): metadata filter, near-duplicate collapsing, candidate
pool size, neighbour expansion for context.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter, defaultdict
from typing import Any

from .day6.ingestion import flatten

STOPWORDS = set("""a an the of to in on for and or is are was be been it its this that these those with as by at from
what which who whom how why when where does do did can could should would will may might must i my me we you your
not no if then than there their them they so such into out about after before over under also only just any all
""".split())
_TOKEN = re.compile(r"[a-z_][a-z0-9_]*|\d+")


def tokenize(text: str) -> list[str]:
    """Lowercase word/identifier tokens. Identifiers with underscores are kept
    whole (``__type_params__``) and also split into parts (``type``, ``params``)."""
    out = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        out.append(tok)
        if "_" in tok:
            out.extend(p for p in tok.split("_") if p and p not in STOPWORDS)
    return out


def _matches(meta: dict, filters: dict | None) -> bool:
    return not filters or all(str(meta.get(k)) == str(v) for k, v in filters.items())


def _result(chunk: dict, score: float, rank: int, method: str, **extra) -> dict:
    return {"id": chunk["id"], "text": chunk["text"], "source_id": chunk["source_id"], "source": chunk["source"],
            "page": chunk["page"], "metadata": chunk["metadata"], "score": round(float(score), 6),
            "rank": rank, "method": method, **extra}


class DenseRetriever:
    name = "dense"

    def __init__(self, store, embedder, collection: str, chunks_by_id: dict[str, dict]):
        self.store, self.embedder, self.collection, self.by_id = store, embedder, collection, chunks_by_id
        if store.dimensions != embedder.dimensions:
            raise ValueError("query embedding dimension does not match the index")

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3) -> list[dict]:
        vec = self.embedder.embed_query(query)
        hits = self.store.search(self.collection, vec, top_k, filters)
        return [_result(self.by_id[h["chunk_id"]], h["score"], i, self.name) for i, h in enumerate(hits, 1)]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tf = [Counter(tokenize(c["text"])) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(term for t in self.tf for term in t)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3) -> list[dict]:
        q = tokenize(query)
        scores = []
        for i, c in enumerate(self.chunks):
            if not _matches(c["metadata"], filters):
                continue
            tf, s = self.tf[i], 0.0
            for term in q:
                f = tf.get(term)
                if f:
                    s += self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if s > 0:
                scores.append((s, i))
        scores.sort(key=lambda x: (-x[0], self.chunks[x[1]]["id"]))
        return [_result(self.chunks[i], s, r, self.name) for r, (s, i) in enumerate(scores[:top_k], 1)]


class HybridRetriever:
    """RRF: score = sum_m 1 / (k + rank_m). Ranks, not raw scores, are fused, so
    cosine and BM25 scales never need to be compared."""
    name = "hybrid_rrf"

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, pool: int = 30, k: int = 60):
        self.dense, self.bm25, self.pool, self.k = dense, bm25, pool, k

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3) -> list[dict]:
        lists = {"dense": self.dense.search(query, filters=filters, top_k=self.pool),
                 "bm25": self.bm25.search(query, filters=filters, top_k=self.pool)}
        fused: dict[str, float] = defaultdict(float)
        ranks: dict[str, dict] = defaultdict(dict)
        chunk_of = {}
        for m, rows in lists.items():
            for r in rows:
                fused[r["id"]] += 1.0 / (self.k + r["rank"])
                ranks[r["id"]][m] = r["rank"]
                chunk_of[r["id"]] = r
        order = sorted(fused, key=lambda i: (-fused[i], i))[:top_k]
        return [_result(chunk_of[i], fused[i], n, self.name, component_ranks=ranks[i]) for n, i in enumerate(order, 1)]


def body_key(chunk: dict) -> str:
    """Duplicate key: chunk text without Day 6's context line, whitespace-flattened.
    Identical paragraphs repeated in 3.11/3.13 or in topic 'compound' share a key."""
    text = chunk["text"]
    if chunk["metadata"].get("heading_prefixed") and text.startswith("[") and "\n" in text:
        text = text.split("\n", 1)[1]
    return hashlib.sha1(flatten(text).encode("utf-8")).hexdigest()


class DedupRetriever:
    """Wrap a retriever: over-fetch, collapse chunks with identical body text,
    keep the best-ranked copy and list the collapsed copies (still traceable)."""

    def __init__(self, base, overfetch: int = 4):
        self.base, self.overfetch = base, overfetch
        self.name = base.name + "+dedup"

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3) -> list[dict]:
        rows = self.base.search(query, filters=filters, top_k=top_k * self.overfetch)
        seen: dict[str, dict] = {}
        for r in rows:
            k = body_key(r)
            if k in seen:
                seen[k].setdefault("duplicates", []).append(r["id"])
            else:
                seen[k] = dict(r)
        out = list(seen.values())[:top_k]
        for i, r in enumerate(out, 1):
            r["rank"], r["method"] = i, self.name
        return out


class NeighborExpander:
    """Context helper: add the previous/next chunk of the same page after a hit
    (marked ``expanded_from``) so evidence cut at a chunk boundary is recovered."""

    def __init__(self, chunks: list[dict]):
        self.by_page: dict[tuple, list[dict]] = defaultdict(list)
        for c in chunks:
            self.by_page[(c["source_id"], c["page"])].append(c)
        for v in self.by_page.values():
            v.sort(key=lambda c: c["metadata"]["chunk_index"])

    def neighbours(self, chunk: dict) -> list[dict]:
        page = self.by_page[(chunk["source_id"], chunk["page"])]
        idx = chunk["metadata"]["chunk_index"]
        return [dict(c, expanded_from=chunk["id"], score=None) for c in page
                if abs(c["metadata"]["chunk_index"] - idx) == 1]


def pack_context(results: list[dict], budget: int, expander: NeighborExpander | None = None) -> tuple[str, list[dict]]:
    """Greedy packing in rank order under a character budget (header included).
    Duplicate IDs are skipped; with an expander, neighbours follow their hit."""
    picked, ids, used = [], set(), 0
    queue = []
    for r in results:
        queue.append(r)
        if expander:
            queue.extend(expander.neighbours(r))
    for r in queue:
        if r["id"] in ids:
            continue
        block = format_chunk(r)
        if used + len(block) > budget:
            continue
        picked.append(r)
        ids.add(r["id"])
        used += len(block)
    return "\n\n".join(format_chunk(r) for r in picked), picked


def format_chunk(r: dict) -> str:
    m = r["metadata"]
    return (f"[{r['id']}] (source: {r['source']}, document: {r['source_id']}, page: {r['page']}, "
            f"python_version: {m.get('python_version')}, section: {m.get('section')})\n{r['text']}")


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, (time.perf_counter() - t0) * 1000
