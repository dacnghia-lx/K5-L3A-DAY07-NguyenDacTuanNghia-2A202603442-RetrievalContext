"""Rebuild the Day 6 index inside this repo and export data/chunks.json.

Same chunks/model/collection as Day 6 (see src/corpus.py). Idempotent: the
Day 6 ChromaStore.sync only embeds new/changed chunks and deletes obsolete ones.
Run: python -m scripts.build_index
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.corpus import CHROMA_DIR, CHUNKS_JSON, COLLECTION, RESULTS, build_chunks, load_pages  # noqa: E402
from src.day6.chunking import by_paragraph  # noqa: E402
from src.day6.embeddings import WordLlamaEmbedder  # noqa: E402
from src.day6.store import ChromaStore  # noqa: E402


def main():
    errors = []
    pages = load_pages(errors)
    chunks = build_chunks(pages)
    CHUNKS_JSON.write_text(json.dumps(chunks, ensure_ascii=False, indent=0), encoding="utf-8")
    emb = WordLlamaEmbedder()
    store = ChromaStore(CHROMA_DIR, emb.dimensions)
    day6_chunks = [c for p in pages for c in by_paragraph(p, 1000, "B", True)]
    t0 = time.perf_counter()
    first = store.sync(COLLECTION, day6_chunks, emb.embed_documents)
    t1 = time.perf_counter()
    second = store.sync(COLLECTION, day6_chunks, emb.embed_documents)
    info = {"pages": len(pages), "chunks": len(chunks), "errors": errors, "collection": COLLECTION,
            "embedding": {"model": emb.model_name, "version": emb.version, "dimensions": emb.dimensions},
            "chroma_version": store.version, "sync": first, "sync_seconds": round(t1 - t0, 3),
            "reingest_check": second}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "index_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    if second["embedded"] or second["deleted"]:
        sys.exit("re-ingest check failed")


if __name__ == "__main__":
    main()
