"""Run the real tool-calling agent on a split for one or more agent configs.

Run (repo root, .env with LLM_API_KEY):
  python -m scripts.run_agent --split dev --configs A1,A2,A3
  python -m scripts.run_agent --split test --configs A1,A2
Resumable: queries already in traces.json are skipped (use --fresh to redo).
Outputs per config: results/agent/<split>/<config>/traces.json (starter
agent_metrics format + full events), cases.json, auto_summary.json.
Only the question text goes to the agent; gold is used afterwards for the
automatic Context Evidence Rate only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent import run_agent  # noqa: E402
from src.corpus import RESULTS, chunk_has_evidence, load_benchmark  # noqa: E402
from src.pipeline import CONFIGS, build_retriever, load_components  # noqa: E402

AGENT_CONFIGS = {
    "A1": {"retrieval": "R1", "max_searches": 3, "context_k": 3, "budget": 8000,
           "description": "Agent + dense Day 6 (baseline)"},
    "A2": {"retrieval": "R5", "max_searches": 3, "context_k": 5, "budget": 8000,
           "description": "Agent + hybrid RRF + dedup + neighbours (retrieval đã chọn)"},
    "A3": {"retrieval": "R5", "max_searches": 1, "context_k": 5, "budget": 8000,
           "description": "Như A2 nhưng chỉ 1 lượt search (ablation multi-turn)"},
    # Added after reading the dev A1-A3 traces: most false refusals were answers whose
    # citations were in `citations` but not written as [id] in the text.
    "A4": {"retrieval": "R5", "max_searches": 3, "context_k": 5, "budget": 8000, "citation_repairs": 1,
           "description": "A2 + 1 lần sửa citation khi final_answer bị từ chối"},
    "A5": {"retrieval": "R1", "max_searches": 3, "context_k": 3, "budget": 8000, "citation_repairs": 1,
           "description": "A1 (dense) + 1 lần sửa citation (tách tác động của retrieval và của citation repair)"},
}


def auto_summary(cases, traces, by_id):
    rows = [t for t in traces if t["status"] != "error"]
    ans_cases = {c["id"]: c for c in cases if c["answerable"]}
    ctx_ev = 0
    for t in rows:
        if t["query_id"] in ans_cases:
            ctx = [by_id[i] for i in t["final_context_ids"]]
            t["context_has_evidence"] = any(chunk_has_evidence(c, ans_cases[t["query_id"]]["gold"]) for c in ctx)
            ctx_ev += t["context_has_evidence"]
    n_ans = sum(t["query_id"] in ans_cases for t in rows)
    searches = [t["n_searches"] for t in rows]
    return {"runs": len(traces), "errors": sum(t["status"] == "error" for t in traces),
            "status": {s: sum(t["status"] == s for t in rows) for s in ("answered", "abstain")},
            "last_event": {k: sum(t["events"][-1]["event"] == k for t in rows if t["events"])
                           for k in sorted({t["events"][-1]["event"] for t in rows if t["events"]})},
            "citation_repair_requests": sum(e["event"] == "citation_repair_request" for t in rows for e in t["events"]),
            "tool_use": f"{sum(n > 0 for n in searches)}/{len(rows)}",
            "searches_distribution": {str(k): searches.count(k) for k in sorted(set(searches))},
            "context_evidence_rate": f"{ctx_ev}/{n_ans}",
            "answered_on_answerable": sum(t["status"] == "answered" for t in rows if t["query_id"] in ans_cases),
            "abstain_on_unanswerable": sum(t["status"] == "abstain" for t in rows if t["query_id"] not in ans_cases),
            "prompt_tokens": sum(t["usage"]["prompt_tokens"] for t in rows),
            "completion_tokens": sum(t["usage"]["completion_tokens"] for t in rows),
            "cost_usd": round(sum(t["cost_usd"] for t in rows), 6),
            "latency_s_mean": round(statistics.mean(t["latency_s"] for t in rows), 2) if rows else None,
            "avg_final_context_chars": round(statistics.mean(t["final_context_chars"] for t in rows)) if rows else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--configs", default="A1,A2,A3")
    ap.add_argument("--ids", default="", help="comma-separated subset of query IDs")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between questions (free-tier rate limits)")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    from src.llm import OpenAICompatibleLLM
    llm = OpenAICompatibleLLM()
    comps = load_components()
    cases = [q for q in load_benchmark() if q["split"] == args.split]
    if args.ids:
        keep = set(args.ids.split(","))
        cases = [q for q in cases if q["id"] in keep]
    for name in args.configs.split(","):
        acfg = AGENT_CONFIGS[name]
        retriever, expander = build_retriever(CONFIGS[acfg["retrieval"]], comps)
        out = RESULTS / "agent" / args.split / name
        out.mkdir(parents=True, exist_ok=True)
        tpath = out / "traces.json"
        traces = [] if args.fresh or not tpath.exists() else json.loads(tpath.read_text(encoding="utf-8"))
        done = {t["query_id"] for t in traces if t["status"] != "error"}
        traces = [t for t in traces if t["query_id"] in done]
        for q in cases:
            if q["id"] in done:
                continue
            print(f"[{name}] {q['id']} ...", flush=True)
            try:
                t = run_agent(q["question"], llm, retriever, expander=expander, max_searches=acfg["max_searches"],
                              context_k=acfg["context_k"], budget=acfg["budget"], query_id=q["id"],
                              citation_repairs=acfg.get("citation_repairs", 0))
            except Exception as e:  # provider / store errors are recorded, never faked
                t = {"query_id": q["id"], "status": "error", "error": f"{type(e).__name__}: {e}",
                     "traceback": traceback.format_exc(limit=3), "events": [], "citations": []}
            t["config"] = name
            t["model"] = llm.model
            t["question"] = q["question"]
            traces.append(t)
            tpath.write_text(json.dumps(traces, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"    -> {t['status']} {t.get('answer', t.get('error', ''))[:150]!r}", flush=True)
            if args.sleep:
                time.sleep(args.sleep)
        all_cases = [q for q in load_benchmark() if q["split"] == args.split]
        (out / "cases.json").write_text(json.dumps(
            [{"id": q["id"], "split": q["split"], "answerable": q["answerable"]} for q in all_cases
             if q["id"] in {t["query_id"] for t in traces}], ensure_ascii=False, indent=1), encoding="utf-8")
        summ = {"config": name, **acfg, "model": llm.model, "base_url": llm.base_url,
                **auto_summary(all_cases, [t for t in traces if t["status"] != "error"] + [t for t in traces if t["status"] == "error"], comps["by_id"])}
        tpath.write_text(json.dumps(traces, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / "auto_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
