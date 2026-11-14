"""Fill `gold` for answerable questions (every chunk-page containing an evidence
quote, restricted to the user's python_version filter) and check that
unanswerable questions' key terms do not occur. No retrieval is run.
Run: python -m scripts.verify_benchmark
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.corpus import BENCHMARK, load_chunks  # noqa: E402
from src.day6.ingestion import flatten  # noqa: E402

ABSENT_TERMS = {
    "du1": ["GIL", "global interpreter lock", "free-thread", "free thread"],
    "du2": ["getsizeof"],
    "du3": ["threshold"],
    "tu1": ["recursion limit", "getrecursionlimit", "setrecursionlimit"],
    "tu2": ["max_str_digits", "4300", "int_max_str"],
}


def main():
    doc = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    chunks = load_chunks()
    flat = [(c, flatten(c["text"])) for c in chunks]
    corpus_lower = " ".join(f for _, f in flat).lower()
    bad = []
    for q in doc["questions"]:
        if not q["answerable"]:
            hits = [t for t in ABSENT_TERMS[q["id"]] if (t.lower() in corpus_lower if t != "GIL" else " GIL" in " ".join(f for _, f in flat))]
            q["absence_check"] = {"terms": ABSENT_TERMS[q["id"]], "found": hits}
            if hits:
                bad.append(f"{q['id']}: terms found {hits}")
            continue
        version = (q.get("filter") or {}).get("python_version")
        gold = {}
        for ev in q["evidence_quotes"]:
            for c, f in flat:
                if flatten(ev) in f and (version is None or c["metadata"]["python_version"] == version):
                    gold[(c["source_id"], c["page"], ev)] = {"source_id": c["source_id"], "page": c["page"], "evidence": ev}
        q["gold"] = sorted(gold.values(), key=lambda g: (g["source_id"], g["page"]))
        if not any(g["source_id"] == q["gold_source_id_primary"] for g in q["gold"]):
            bad.append(f"{q['id']}: primary source has no chunk containing the evidence")
        print(f"{q['id']:6} {q['split']:4} gold pages={len({(g['source_id'], g['page']) for g in q['gold']})}")
    if bad:
        print("\n".join(bad)); sys.exit(1)
    BENCHMARK.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("OK")


if __name__ == "__main__":
    main()
