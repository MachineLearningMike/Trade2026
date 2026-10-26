from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from typing import Any, Dict

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from fed_agents.fed_data_agent import build_fed_data_graph
from fed_agents.model_research_agent import build_model_research_graph


def run_parallel_agents() -> Dict[str, Any]:
    shared: Dict[str, Any] = {"ready": threading.Event(), "fed_data": None, "fed_summary": None}

    def collect_worker() -> None:
        graph = build_fed_data_graph()
        result = graph.invoke({"series_id": "FEDFUNDS"})
        shared["fed_data"] = result["fed_data"]
        shared["fed_summary"] = result["fed_summary"]
        shared["ready"].set()

    def research_worker() -> None:
        shared["ready"].wait(timeout=60)
        if shared["fed_data"] is None:
            raise RuntimeError("FED data collection did not complete in time.")
        graph = build_model_research_graph()
        result = graph.invoke({
            "fed_data": shared["fed_data"],
            "fed_summary": shared["fed_summary"],
        })
        shared["research_result"] = result

    t1 = threading.Thread(target=collect_worker, name="fed_data_agent", daemon=True)
    t2 = threading.Thread(target=research_worker, name="research_agent", daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    return {
        "fed_summary": shared.get("fed_summary"),
        "fed_data": shared.get("fed_data"),
        "research_result": shared.get("research_result"),
    }


async def run_parallel_agents_async() -> Dict[str, Any]:
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, run_parallel_agents)
    return result


if __name__ == "__main__":
    result = run_parallel_agents()
    print("\n=== FED DATA SUMMARY ===")
    print(result["fed_summary"])
    print("\n=== MODEL RESEARCH REPORT ===")
    print(result["research_result"]["report"])
