"""Offline checks for the Day 7 code (BM25, RRF, dedup, agent loop with a scripted
fake LLM). Fake components only test behaviour; they never produce benchmark numbers."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent import run_agent  # noqa: E402
from src.retrieval import BM25Retriever, DedupRetriever, HybridRetriever, NeighborExpander, pack_context, tokenize  # noqa: E402


def chunk(cid, text, version="3.13", page=1, idx=0, source_id="py3.13/try"):
    return {"id": cid, "text": text, "source_id": source_id, "source": "data/raw/x.py", "page": page,
            "metadata": {"python_version": version, "chunk_index": idx, "section": "s", "heading_prefixed": False}}


CHUNKS = [
    chunk("c1", "The finally clause is executed; the saved exception is discarded by return.", idx=0),
    chunk("c2", "The finally clause is executed; the saved exception is discarded by return.", "3.11", source_id="py3.11/try"),
    chunk("c3", "A lambda cannot contain statements or annotations.", source_id="py3.13/lambda"),
    chunk("c4", "Next paragraph of the try page about else.", idx=1),
]


class FakeDense:
    name = "dense"
    def search(self, query, *, filters=None, top_k=3):
        rows = [c for c in CHUNKS if not filters or c["metadata"]["python_version"] == filters["python_version"]]
        return [dict(c, score=1.0 / (i + 1), rank=i + 1, method="dense") for i, c in enumerate(rows[:top_k])]


class ScriptedLLM:
    def __init__(self, turns): self.turns = list(turns); self.seen = []
    def chat(self, messages, tools):
        self.seen.append(messages[-1])
        calls = self.turns.pop(0)
        return {"content": "", "tool_calls": [{"id": f"t{i}", "name": n, "arguments": json.dumps(a)} for i, (n, a) in enumerate(calls)],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "cost_usd": 0.0, "latency_ms": 1}


class RetrievalTests(unittest.TestCase):
    def test_tokenizer_keeps_identifiers(self):
        self.assertIn("__type_params__", tokenize("the __type_params__ attribute"))
        self.assertIn("params", tokenize("__type_params__"))
        self.assertNotIn("the", tokenize("the try"))

    def test_bm25_filter_and_rank(self):
        r = BM25Retriever(CHUNKS)
        rows = r.search("lambda statements", top_k=2)
        self.assertEqual(rows[0]["id"], "c3")
        rows = r.search("finally discarded", filters={"python_version": "3.11"}, top_k=3)
        self.assertEqual([x["id"] for x in rows], ["c2"])

    def test_hybrid_rrf_fuses_ranks(self):
        h = HybridRetriever(FakeDense(), BM25Retriever(CHUNKS), pool=4)
        rows = h.search("lambda annotations", top_k=3)
        self.assertEqual(len({x["id"] for x in rows}), len(rows))
        self.assertIn("component_ranks", rows[0])

    def test_dedup_collapses_identical_bodies(self):
        rows = DedupRetriever(FakeDense()).search("finally", top_k=3)
        ids = [x["id"] for x in rows]
        self.assertEqual(ids[0], "c1")
        self.assertNotIn("c2", ids)
        self.assertEqual(rows[0]["duplicates"], ["c2"])

    def test_pack_context_budget_and_neighbours(self):
        exp = NeighborExpander(CHUNKS)
        text, picked = pack_context([CHUNKS[0]], 10_000, exp)
        self.assertEqual([p["id"] for p in picked], ["c1", "c4"])
        text, picked = pack_context(CHUNKS, 150)
        self.assertLessEqual(len(text), 150)


class AgentTests(unittest.TestCase):
    def test_search_then_cited_answer(self):
        llm = ScriptedLLM([[("search", {"query": "finally return", "python_version": "3.13"})],
                           [("final_answer", {"answer": "It is discarded [c1].", "citations": ["c1"], "abstain": False})]])
        out = run_agent("q?", llm, FakeDense(), query_id="x")
        self.assertEqual(out["status"], "answered")
        self.assertEqual(out["citations"], ["c1"])
        s = [e for e in out["events"] if e["event"] == "search"][0]
        self.assertEqual(s["filter"], {"python_version": "3.13"})
        self.assertNotIn("c2", out["final_context_ids"])

    def test_citation_not_in_context_is_abstention(self):
        llm = ScriptedLLM([[("search", {"query": "q"})],
                           [("final_answer", {"answer": "Yes [zzz:1].", "citations": ["zzz:1"], "abstain": False})]])
        self.assertEqual(run_agent("q?", llm, FakeDense())["status"], "abstain")

    def test_citation_repair_off_by_default(self):
        llm = ScriptedLLM([[("search", {"query": "finally"})],
                           [("final_answer", {"answer": "It is discarded.", "citations": ["c1"], "abstain": False})]])
        out = run_agent("q?", llm, FakeDense())
        self.assertEqual(out["status"], "abstain")
        self.assertEqual(out["events"][-1]["event"], "invalid_or_missing_citations")

    def test_citation_repair_lets_model_fix_inline_ids(self):
        llm = ScriptedLLM([[("search", {"query": "finally"})],
                           [("final_answer", {"answer": "It is discarded.", "citations": ["c1"], "abstain": False})],
                           [("final_answer", {"answer": "It is discarded [c1].", "citations": ["c1"], "abstain": False})]])
        out = run_agent("q?", llm, FakeDense(), citation_repairs=1)
        self.assertEqual(out["status"], "answered")
        self.assertEqual(out["citations"], ["c1"])
        self.assertIn("[c1]", llm.seen[-1]["content"])  # the rejection told the model which ID to inline
        self.assertEqual(sum(e["event"] == "citation_repair_request" for e in out["events"]), 1)

    def test_citation_repair_is_bounded(self):
        bad = ("final_answer", {"answer": "Yes [zzz:1].", "citations": ["zzz:1"], "abstain": False})
        llm = ScriptedLLM([[("search", {"query": "q"})], [bad], [bad]])
        out = run_agent("q?", llm, FakeDense(), citation_repairs=1)
        self.assertEqual(out["status"], "abstain")
        self.assertEqual(out["events"][-1]["event"], "invalid_or_missing_citations")

    def test_explicit_abstain(self):
        llm = ScriptedLLM([[("search", {"query": "gil"})],
                           [("final_answer", {"answer": "Not in the docs.", "citations": [], "abstain": True})]])
        out = run_agent("GIL?", llm, FakeDense())
        self.assertEqual(out["status"], "abstain")
        self.assertEqual(out["n_searches"], 1)

    def test_search_limit(self):
        llm = ScriptedLLM([[("search", {"query": "a"})], [("search", {"query": "b"})],
                           [("final_answer", {"answer": "x", "citations": [], "abstain": True})]])
        out = run_agent("q", llm, FakeDense(), max_searches=1)
        self.assertEqual(out["n_searches"], 1)
        self.assertTrue(any(e["event"] == "search_limit" for e in out["events"]))

    def test_budget_is_respected(self):
        llm = ScriptedLLM([[("search", {"query": "a"})],
                           [("final_answer", {"answer": "x", "citations": [], "abstain": True})]])
        out = run_agent("q", llm, FakeDense(), budget=200)
        self.assertLessEqual(out["final_context_chars"], 200)

    def test_retriever_errors_propagate(self):
        class Broken:
            def search(self, *a, **k): raise ConnectionError("store down")
        llm = ScriptedLLM([[("search", {"query": "a"})]])
        with self.assertRaises(ConnectionError):
            run_agent("q", llm, Broken())


if __name__ == "__main__":
    unittest.main()
