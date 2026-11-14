"""Benchmark retrieval methods/configs (no LLM): Hit@1, Evidence Hit@3, MRR@10,
Context Evidence Rate, latency. Query = original question text; filter = the
user-stated python_version (benchmark `filter`), only for configs with filter.

Run: python -m scripts.eval_retrieval --split dev [--configs R1,R3]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.corpus import RESULTS, chunk_has_evidence, load_benchmark  # noqa: E402
from src.pipeline import CONFIGS, build_retriever  # noqa: E402
from src.retrieval import pack_context, timed  # noqa: E402


def evaluate(cfg_name: str, split: str, questions, components) -> dict:
    cfg = CONFIGS[cfg_name]
    retriever, expander = build_retriever(cfg, components)
    rows, lat = [], []
    for q in questions:
        if q["split"] != split:
            continue
        flt = q.get("filter") if cfg["filter"] else None
        ranked, ms = timed(retriever.search, q["question"], filters=flt, top_k=max(10, cfg["context_k"]))
        lat.append(ms)
        top3 = ranked[:3]
        context_text, ctx = pack_context(ranked[:cfg["context_k"]], cfg["budget"], expander)
        row = {"query_id": q["id"], "split": split, "answerable": q["answerable"], "question": q["question"],
               "query_used": q["question"], "filter_applied": flt, "latency_ms": round(ms, 2),
               "top3": [{k: r[k] for k in ("id", "rank", "score", "source_id", "page", "method")}
                        | {"python_version": r["metadata"].get("python_version"),
                           "component_ranks": r.get("component_ranks"), "duplicates": r.get("duplicates"),
                           "text": r["text"]} for r in top3],
               "context_ids": [r["id"] for r in ctx], "context_chars": len(context_text)}
        if q["answerable"]:
            hits = [chunk_has_evidence(r, q["gold"]) for r in ranked]
            row.update({"hit_at_1": bool(hits[:1] and hits[0]), "hit_at_3": any(hits[:3]),
                        "first_hit_rank": hits.index(True) + 1 if True in hits else None,
                        "context_has_evidence": any(chunk_has_evidence(r, q["gold"]) for r in ctx),
                        "gold_pages": sorted({f"{g['source_id']}#{g['page']}" for g in q["gold"]})})
        rows.append(row)
    ans = [r for r in rows if r["answerable"]]
    n = len(ans)
    mrr = sum(1 / r["first_hit_rank"] for r in ans if r["first_hit_rank"] and r["first_hit_rank"] <= 10) / n
    summary = {"config": cfg_name, "description": cfg["description"], "split": split, "N": n,
               "hit1": sum(r["hit_at_1"] for r in ans), "hit3": sum(r["hit_at_3"] for r in ans),
               "context_evidence": sum(r["context_has_evidence"] for r in ans), "mrr10": round(mrr, 3),
               "avg_context_chars": round(statistics.mean(r["context_chars"] for r in ans)),
               "latency_ms_mean": round(statistics.mean(lat), 2), "cost_usd": 0.0,
               "misses_at_3": [r["query_id"] for r in ans if not r["hit_at_3"]],
               "context_misses": [r["query_id"] for r in ans if not r["context_has_evidence"]]}
    return {"summary": summary, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    args = ap.parse_args()
    from src.pipeline import load_components
    comps = load_components()
    qs = load_benchmark()
    out_dir = RESULTS / "retrieval" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    table = []
    for name in args.configs.split(","):
        res = evaluate(name, args.split, qs, comps)
        (out_dir / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        s = res["summary"]
        table.append(s)
        print(f"{name:4} {args.split} Hit@1 {s['hit1']}/{s['N']} Hit@3 {s['hit3']}/{s['N']} "
              f"CtxEv {s['context_evidence']}/{s['N']} MRR {s['mrr10']} ctx {s['avg_context_chars']}c "
              f"{s['latency_ms_mean']}ms misses {s['misses_at_3']}")
    (out_dir / "summary.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    lines = ["| Config | Mô tả | Hit@1 | Evidence Hit@3 | Context Evidence Rate | MRR@10 | Context TB (ký tự) | Latency TB | Chi phí |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in table:
        f = lambda k: f"{s[k]}/{s['N']} ({s[k] / s['N']:.0%})"
        lines.append(f"| {s['config']} | {s['description']} | {f('hit1')} | {f('hit3')} | {f('context_evidence')} | "
                     f"{s['mrr10']} | {s['avg_context_chars']} | {s['latency_ms_mean']} ms | 0 |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
