from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, TypedDict

import pandas as pd
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from config import get_config, resolve_data_dir
from macro_01_downloader import FSVDownloader
from macro_02_find_value_intervals import build_availability
from macro_dictionary import FEATURES, get_features


class FedDataAgentState(TypedDict, total=False):
    config: Dict[str, Any]
    feature_name: str
    feature_meta: Dict[str, Any]
    raw_df: pd.DataFrame
    raw_path: str
    intervals_df: pd.DataFrame
    intervals_path: str
    summary: str
    status: str


def _normalize_legacy_macro_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()
    for c in list(df.columns):
        if c.endswith("_x") or c.endswith("_y"):
            base = c[:-2]
            if base not in df.columns:
                df = df.rename(columns={c: base})
            else:
                df = df.drop(columns=[c])
    return df


def _resolve_api_keys(config: Dict[str, Any]) -> tuple[str | None, str | None]:
    fred = os.environ.get("FRED_API_KEY") or config.get("fred_api_key") or None
    twelve = os.environ.get("TWELVEDATA_API_KEY") or config.get("twelvedata_api_key") or None
    return fred, twelve


def load_feature_meta(state: FedDataAgentState) -> Dict[str, Any]:
    cfg = state.get("config") or get_config()
    catalog = get_features(cfg, apply_filters=False)
    feature_name = "fed_funds"
    if feature_name not in catalog:
        raise KeyError(f"Feature '{feature_name}' was not found in the macro catalog.")

    state["feature_name"] = feature_name
    state["feature_meta"] = catalog[feature_name]
    state["config"] = cfg
    state["status"] = "catalog_loaded"
    return {"feature_name": feature_name, "feature_meta": catalog[feature_name], "status": "catalog_loaded"}


def download_fed_data(state: FedDataAgentState) -> Dict[str, Any]:
    cfg = state["config"]
    feature_name = state["feature_name"]
    meta = state["feature_meta"]
    data_dir = Path(resolve_data_dir(cfg.get("data_dir", "dataset_cache")))
    data_dir.mkdir(parents=True, exist_ok=True)

    fred_key, twelve_key = _resolve_api_keys(cfg)
    if not fred_key:
        raise RuntimeError(
            "FRED API key required. Set env FRED_API_KEY or config['fred_api_key']."
        )

    downloader = FSVDownloader(
        fred_api_key=fred_key,
        twelvedata_api_key=twelve_key,
        features_catalog={feature_name: meta},
    )

    raw_path = data_dir / "macro_raw.csv"
    existing = None
    if raw_path.exists():
        existing = pd.read_csv(raw_path, parse_dates=["date"])
        existing = _normalize_legacy_macro_columns(existing)
        for c in existing.columns:
            if c.endswith("__first_release"):
                existing[c] = pd.to_datetime(existing[c], errors="coerce")

    merged = downloader.download_all(
        start_date="1990-01-01",
        end_date=None,
        features_catalog={feature_name: meta},
        existing_df=existing,
        skip_existing=True,
    )
    merged = merged.sort_values("date").reset_index(drop=True)
    merged.to_csv(raw_path, index=False, float_format="%.8g")

    state["raw_df"] = merged
    state["raw_path"] = str(raw_path)
    state["status"] = "downloaded"
    print(f"Saved FED raw data to {raw_path}")
    return {"raw_df": merged, "raw_path": str(raw_path), "status": "downloaded"}


def build_fed_intervals(state: FedDataAgentState) -> Dict[str, Any]:
    cfg = state["config"]
    feature_name = state["feature_name"]
    meta = state["feature_meta"]
    data_dir = Path(resolve_data_dir(cfg.get("data_dir", "dataset_cache")))
    raw_path = Path(state["raw_path"]) if state.get("raw_path") else data_dir / "macro_raw.csv"

    raw = pd.read_csv(raw_path, parse_dates=["date"])
    raw = _normalize_legacy_macro_columns(raw)
    for c in raw.columns:
        if c.endswith("__first_release"):
            raw[c] = pd.to_datetime(raw[c], errors="coerce")

    catalog = {feature_name: meta}
    intervals = build_availability(raw, catalog)
    out_path = data_dir / "macro_value_intervals.csv"
    intervals.to_csv(out_path, index=False, float_format="%.8g")

    state["intervals_df"] = intervals
    state["intervals_path"] = str(out_path)
    state["status"] = "processed"
    print(f"Saved FED intervals to {out_path}")
    return {"intervals_df": intervals, "intervals_path": str(out_path), "status": "processed"}


def summarize_fed_data(state: FedDataAgentState) -> Dict[str, Any]:
    raw = state["raw_df"]
    if raw.empty or "date" not in raw.columns or "fed_funds" not in raw.columns:
        raise ValueError("No FED raw data available to summarize.")

    df = raw[["date", "fed_funds"]].dropna().copy()
    df["date"] = pd.to_datetime(df["date"])
    summary = (
        f"FED data processed for fed_funds from {df['date'].min().date()} "
        f"to {df['date'].max().date()}. "
        f"Rows={len(df)}, mean={df['fed_funds'].mean():.4f}, std={df['fed_funds'].std(ddof=0):.4f}."
    )
    state["summary"] = summary
    state["status"] = "summarized"
    return {"summary": summary, "status": "summarized", "messages": [AIMessage(content=summary)]}


def build_fed_graph():
    graph = StateGraph(FedDataAgentState)
    graph.add_node("load_feature_meta", load_feature_meta)
    graph.add_node("download_fed_data", download_fed_data)
    graph.add_node("build_fed_intervals", build_fed_intervals)
    graph.add_node("summarize_fed_data", summarize_fed_data)

    graph.set_entry_point("load_feature_meta")
    graph.add_edge("load_feature_meta", "download_fed_data")
    graph.add_edge("download_fed_data", "build_fed_intervals")
    graph.add_edge("build_fed_intervals", "summarize_fed_data")
    graph.add_edge("summarize_fed_data", END)
    return graph.compile()


if __name__ == "__main__":
    graph = build_fed_graph()
    result = graph.invoke({"config": get_config()})
    print("\n=== FED DATA AGENT RESULT ===")
    print(result["summary"])
    print(f"Raw CSV: {result.get('raw_path')}")
    print(f"Intervals CSV: {result.get('intervals_path')}")
