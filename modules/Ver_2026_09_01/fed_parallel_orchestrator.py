from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Dict

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

module_root = Path(__file__).resolve().parent
if str(module_root) not in sys.path:
    sys.path.insert(0, str(module_root))

from config import get_config
from fed_data_agent import build_fed_graph
from fed_model_agent import build_fed_model_graph


def run_fed_parallel_agents() -> Dict[str, Any]:
    shared: Dict[str, Any] = {"ready": threading.Event()}

    def data_worker() -> None:
        graph = build_fed_graph()
        result = graph.invoke({"config": get_config()})
        shared["fed_raw"] = result["raw_df"]
        shared["fed_summary"] = result["summary"]
        shared["raw_path"] = result.get("raw_path")
        shared["intervals_path"] = result.get("intervals_path")
        shared["ready"].set()

    def model_worker() -> None:
        shared["ready"].wait(timeout=120)
        if "fed_raw" not in shared:
            raise RuntimeError("FED data agent did not finish before the model agent started.")
        graph = build_fed_model_graph()
        result = graph.invoke({
            "fed_raw": shared["fed_raw"],
            "fed_intervals": None,
        })
        shared["model_report"] = result["report"]
        shared["model_metrics"] = result["metrics"]
        shared["model_name"] = result["model_name"]

    t1 = threading.Thread(target=data_worker, daemon=True)
    t2 = threading.Thread(target=model_worker, daemon=True)
    t1.start(); t2.start()
    t1.join(); t2.join()

    return {
        "fed_summary": shared.get("fed_summary"),
        "raw_path": shared.get("raw_path"),
        "intervals_path": shared.get("intervals_path"),
        "model_name": shared.get("model_name"),
        "model_metrics": shared.get("model_metrics"),
        "model_report": shared.get("model_report"),
    }


if __name__ == "__main__":
    result = run_fed_parallel_agents()
    print("\n=== FED SUMMARY ===")
    print(result["fed_summary"])
    print("\n=== MODEL REPORT ===")
    print(result["model_report"])
