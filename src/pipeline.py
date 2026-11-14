"""Named retrieval configurations and component loading (shared by the
retrieval benchmark and the agent)."""
from __future__ import annotations

from .corpus import CHROMA_DIR, COLLECTION, load_chunks
from .retrieval import BM25Retriever, DedupRetriever, DenseRetriever, HybridRetriever, NeighborExpander

# context_k = ranked chunks offered to the context packer; budget in characters
CONFIGS = {
    "R0": {"method": "dense", "filter": False, "dedup": False, "pool": None, "neighbors": False,
           "context_k": 3, "budget": 4000, "description": "Dense Day 6 (B), không filter"},
    "R1": {"method": "dense", "filter": True, "dedup": False, "pool": None, "neighbors": False,
           "context_k": 3, "budget": 4000, "description": "Dense Day 6 + filter phiên bản (= Day 6 C) — baseline"},
    "R2": {"method": "bm25", "filter": True, "dedup": False, "pool": None, "neighbors": False,
           "context_k": 3, "budget": 4000, "description": "BM25 (tokenizer giữ định danh) + filter"},
    "R3": {"method": "hybrid", "filter": True, "dedup": False, "pool": 30, "neighbors": False,
           "context_k": 3, "budget": 4000, "description": "Hybrid RRF dense+BM25 (pool 30, k=60) + filter"},
    # improvements (hypotheses written in artifacts/experiment_log.csv before running them)
    "R4": {"method": "hybrid", "filter": True, "dedup": True, "pool": 30, "neighbors": False,
           "context_k": 3, "budget": 4000, "description": "R3 + gộp chunk trùng nội dung (3.11/3.13, compound)"},
    "R5": {"method": "hybrid", "filter": True, "dedup": True, "pool": 30, "neighbors": True,
           "context_k": 5, "budget": 6000, "description": "R4 + context top-5 + chunk lân cận, budget 6000 ký tự"},
    "R6": {"method": "bm25", "filter": True, "dedup": True, "pool": None, "neighbors": False,
           "context_k": 3, "budget": 4000, "description": "R2 + gộp trùng (ablation)"},
}


def load_components() -> dict:
    from .day6.embeddings import WordLlamaEmbedder
    from .day6.store import ChromaStore
    chunks = load_chunks()
    emb = WordLlamaEmbedder()
    store = ChromaStore(CHROMA_DIR, emb.dimensions)
    by_id = {c["id"]: c for c in chunks}
    if store._client.get_collection(COLLECTION).count() != len(chunks):
        raise RuntimeError("Chroma collection and data/chunks.json differ: run python -m scripts.build_index")
    return {"chunks": chunks, "by_id": by_id, "dense": DenseRetriever(store, emb, COLLECTION, by_id),
            "bm25": BM25Retriever(chunks), "embedder": emb, "store": store}


def build_retriever(cfg: dict, comps: dict):
    if cfg["method"] == "dense":
        r = comps["dense"]
    elif cfg["method"] == "bm25":
        r = comps["bm25"]
    elif cfg["method"] == "hybrid":
        r = HybridRetriever(comps["dense"], comps["bm25"], pool=cfg["pool"])
    else:
        raise ValueError(cfg["method"])
    if cfg["dedup"]:
        r = DedupRetriever(r)
    expander = NeighborExpander(comps["chunks"]) if cfg["neighbors"] else None
    return r, expander
