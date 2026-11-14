"""Check the LLM connection: list a few models and make one tool-calling request.
Run: python -m scripts.check_llm
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent import TOOLS  # noqa: E402
from src.llm import OpenAICompatibleLLM  # noqa: E402


def main():
    llm = OpenAICompatibleLLM()
    print("base_url:", llm.base_url, "| model:", llm.model)
    try:
        names = [m.id for m in llm.client.models.list()]
        print("models available (first 40):", names[:40])
    except Exception as e:
        print("models.list failed (not fatal):", type(e).__name__, e)
    resp = llm.chat([{"role": "system", "content": "Call the search tool."},
                     {"role": "user", "content": "What does the pass statement do?"}], TOOLS)
    print(json.dumps({k: resp[k] for k in ("tool_calls", "usage", "latency_ms", "finish_reason")}, indent=1))
    if not resp["tool_calls"]:
        sys.exit("Model answered without a tool call: choose a model that supports tool calling (LLM_MODEL).")
    print("LLM OK")


if __name__ == "__main__":
    main()
