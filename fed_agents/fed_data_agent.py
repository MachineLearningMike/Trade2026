from __future__ import annotations

import os
from typing import Any, Dict, TypedDict

import pandas as pd
import requests
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph


class FedDataState(TypedDict, total=False):
    series_id: str
    source: str
    fed_data: pd.DataFrame
    fed_summary: str
    status: str


def _build_synthetic_fed_data(series_id: str, points: int = 120) -> pd.DataFrame:
    """Generate a small synthetic FRED-like dataset when the network or API key is unavailable."""
    idx = pd.date_range(end=pd.Timestamp.today(), periods=points, freq="MS")
    base = 2.0
    trend = 0.04
    seasonal = 0.35 * pd.Series(range(points), index=idx).map(lambda x: (x % 12) / 12)
    noise = pd.Series([((x * 13) % 17) / 100 for x in range(points)], index=idx)
    values = base + trend * pd.Series(range(points), index=idx) + seasonal + noise
    return pd.DataFrame({"DATE": idx, "value": values.values}).sort_values("DATE").reset_index(drop=True)


def _fetch_fred_data(series_id: str, limit: int = 120) -> pd.DataFrame:
    api_key = os.getenv("FRED_API_KEY")
    params = {
        "series_id": series_id,
        "file_type": "json",
        "limit": limit,
        "sort": "asc",
    }
    if api_key:
        params["api_key"] = api_key
    url = "https://api.stlouisfed.org/fred/series/observations"
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    observations = payload.get("observations", [])
    if not observations:
        raise ValueError(f"No observations returned for {series_id!r}.")
    df = pd.DataFrame(observations)
    df = df[["date", "value"]].copy()
    df["DATE"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values("DATE").reset_index(drop=True)
    return df[["DATE", "value"]].head(limit)


def collect_fed_data(state: FedDataState) -> Dict[str, Any]:
    series_id = state.get("series_id", "FEDFUNDS")
    try:
        data = _fetch_fred_data(series_id, limit=120)
        source = "fred"
    except Exception as exc:
        data = _build_synthetic_fed_data(series_id, points=120)
        source = f"synthetic-fallback ({exc.__class__.__name__}: {exc})"
    state["fed_data"] = data
    state["source"] = source
    state["status"] = "collected"
    return {"fed_data": data, "source": source, "status": "collected"}


def validate_fed_data(state: FedDataState) -> Dict[str, Any]:
    data = state["fed_data"]
    if data is None or data.empty:
        raise ValueError("No FED data was collected.")
    required = {"DATE", "value"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required FED columns: {sorted(missing)}")
    data = data.copy()
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["value"]).reset_index(drop=True)
    state["fed_data"] = data
    state["status"] = "validated"
    return {"fed_data": data, "status": "validated"}


def summarize_fed_data(state: FedDataState) -> Dict[str, Any]:
    data = state["fed_data"]
    first = data.iloc[0]["DATE"].strftime("%Y-%m-%d")
    last = data.iloc[-1]["DATE"].strftime("%Y-%m-%d")
    mean_value = float(data["value"].mean())
    std_value = float(data["value"].std(ddof=0))
    summary = (
        f"Collected {len(data)} observations for {state.get('series_id', 'FEDFUNDS')} "
        f"from {first} to {last}. Mean={mean_value:.3f}, std={std_value:.3f}, "
        f"source={state.get('source', 'unknown')}."
    )
    state["fed_summary"] = summary
    state["status"] = "summarized"
    return {"fed_summary": summary, "status": "summarized", "messages": [AIMessage(content=summary)]}


def build_fed_data_graph():
    graph = StateGraph(FedDataState)
    graph.add_node("collect_fed_data", collect_fed_data)
    graph.add_node("validate_fed_data", validate_fed_data)
    graph.add_node("summarize_fed_data", summarize_fed_data)
    graph.set_entry_point("collect_fed_data")
    graph.add_edge("collect_fed_data", "validate_fed_data")
    graph.add_edge("validate_fed_data", "summarize_fed_data")
    graph.add_edge("summarize_fed_data", END)
    return graph.compile()
