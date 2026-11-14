"""Persistent vector store: Chroma (PersistentClient, on-disk, HNSW, cosine).

The starter suggested pgvector; GUIDE.md allows Chroma as an equivalent store
with persistence, vector search and metadata filtering. One collection per
chunking configuration; strategy C reuses strategy B's collection and only
adds a ``where`` filter, so B and C differ by exactly one factor.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import Chunk

_ALLOWED_FILTERS = {"python_version", "topic", "document_id"}


class ChromaStore:
    def __init__(self, path: Path, dimensions: int) -> None:
        import chromadb
        from chromadb.config import Settings

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.dimensions = dimensions
        self.version = chromadb.__version__
        self._client = chromadb.PersistentClient(
            path=str(self.path), settings=Settings(anonymized_telemetry=False))

    def collection(self, name: str):
        # embedding_function=None: we always pass our own vectors, so Chroma never
        # downloads or runs its default model.
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "hnsw:search_ef": 200,
                      "hnsw:construction_ef": 200, "embedding_dim": self.dimensions},
        )

    def existing(self, name: str) -> dict[str, str]:
        """Return {chunk_id: chunk_sha256} already stored in a collection."""
        col = self.collection(name)
        got = col.get(include=["metadatas"])
        return {i: (m or {}).get("chunk_sha256", "") for i, m in zip(got["ids"], got["metadatas"])}

    def sync(self, name: str, chunks: list[Chunk], embed_fn) -> dict[str, int]:
        """Idempotent ingest: embed only new/changed chunks, delete obsolete ones.

        Stable IDs (strategy:page_id:cNNN) + content hash make re-running the
        same corpus a no-op, and a changed source page replaces its chunks.
        """
        ids = [c.chunk_id for c in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name}: duplicate chunk IDs generated")
        col = self.collection(name)
        before = self.existing(name)
        wanted = {c.chunk_id: c for c in chunks}
        obsolete = [i for i in before if i not in wanted]
        changed = [c for c in chunks if before.get(c.chunk_id) != c.metadata["chunk_sha256"]]
        if obsolete:
            col.delete(ids=obsolete)
        for start in range(0, len(changed), 256):
            batch = changed[start:start + 256]
            vectors = embed_fn([c.text for c in batch])
            if len(vectors) != len(batch) or any(len(v) != self.dimensions for v in vectors):
                raise ValueError(f"{name}: vector count/dimension mismatch")
            col.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=vectors,
                documents=[c.text for c in batch],
                metadatas=[{**c.metadata, "document_id": c.document_id, "source": c.source,
                            "page": c.page} for c in batch],
            )
        return {"before": len(before), "embedded": len(changed), "deleted": len(obsolete),
                "after": col.count()}

    def search(self, name: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if len(query_vector) != self.dimensions:
            raise ValueError("query vector has wrong dimension")
        where = None
        if filters:
            bad = set(filters) - _ALLOWED_FILTERS
            if bad:
                raise ValueError(f"filter fields not allowed: {bad}")
            clauses = [{k: {"$eq": v}} for k, v in filters.items()]
            where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        res = self._client.get_collection(name).query(
            query_embeddings=[query_vector], n_results=top_k, where=where,
            include=["documents", "metadatas", "distances"])
        hits = []
        for rank, (cid, doc, meta, dist) in enumerate(zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1):
            hits.append({"rank": rank, "chunk_id": cid, "score": round(1.0 - float(dist), 6),
                         "cosine_distance": round(float(dist), 6), "text": doc, "metadata": meta})
        return hits

    def all_vectors(self, name: str) -> tuple[list[str], list[list[float]], list[dict[str, Any]]]:
        got = self._client.get_collection(name).get(include=["embeddings", "metadatas"])
        return got["ids"], got["embeddings"], got["metadatas"]
