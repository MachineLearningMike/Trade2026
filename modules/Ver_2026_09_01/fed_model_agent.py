from __future__ import annotations

from typing import Any, Dict, TypedDict

import pandas as pd
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class FedModelAgentState(TypedDict, total=False):
    fed_raw: pd.DataFrame
    fed_intervals: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    model_name: str
    metrics: Dict[str, float]
    report: str
    status: str


def prepare_macro_features(state: FedModelAgentState) -> Dict[str, Any]:
    raw = state["fed_raw"].copy()
    raw = raw.loc[:, ~raw.columns.duplicated()].copy()
    raw = raw.sort_values("date").reset_index(drop=True)
    if "fed_funds" not in raw.columns:
        candidates = [c for c in raw.columns if c.startswith("fed_funds") and c != "fed_funds__first_release"]
        if not candidates:
            raise ValueError("No 'fed_funds' data column found in the FED raw data.")
        raw["fed_funds"] = raw[candidates[0]]

    series = pd.to_numeric(raw["fed_funds"], errors="coerce")
    raw["fed_funds"] = series
    raw["lag_1"] = raw["fed_funds"].shift(1)
    raw["lag_3"] = raw["fed_funds"].shift(3)
    raw["rolling_mean_3"] = raw["fed_funds"].rolling(window=3, min_periods=1).mean().shift(1)
    raw["target"] = raw["fed_funds"].shift(-1)
    model_cols = ["lag_1", "lag_3", "rolling_mean_3", "target"]
    raw = raw[raw[model_cols].notna().all(axis=1)].reset_index(drop=True)
    if raw.empty:
        raise ValueError("FED raw data produced no usable rows after feature generation.")

    X = raw[["lag_1", "lag_3", "rolling_mean_3"]]
    y = raw["target"]
    state["features"] = X
    state["target"] = y
    state["status"] = "features_ready"
    return {"features": X, "target": y, "status": "features_ready"}


def evaluate_fed_models(state: FedModelAgentState) -> Dict[str, Any]:
    X = state["features"]
    y = state["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=42),
    }
    scores = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        scores.append((name, {
            "r2": float(r2_score(y_test, preds)),
            "rmse": float(mse ** 0.5),
            "mae": float(mean_absolute_error(y_test, preds)),
        }))

    best_name, best_metrics = max(scores, key=lambda item: item[1]["r2"])
    state["model_name"] = best_name
    state["metrics"] = best_metrics
    state["status"] = "model_evaluated"
    return {"model_name": best_name, "metrics": best_metrics, "status": "model_evaluated"}


def write_fed_report(state: FedModelAgentState) -> Dict[str, Any]:
    raw = state.get("fed_raw")
    model_name = state.get("model_name", "unknown")
    metrics = state.get("metrics", {})
    rows = len(raw) if raw is not None else 0
    report = (
        f"FED macro research report: {rows} raw observations processed. "
        f"Best model: {model_name}. "
        f"R2={metrics.get('r2', 0.0):.4f}, RMSE={metrics.get('rmse', 0.0):.4f}, MAE={metrics.get('mae', 0.0):.4f}."
    )
    state["report"] = report
    state["status"] = "report_ready"
    return {"report": report, "status": "report_ready", "messages": [AIMessage(content=report)]}


def build_fed_model_graph():
    graph = StateGraph(FedModelAgentState)
    graph.add_node("prepare_macro_features", prepare_macro_features)
    graph.add_node("evaluate_fed_models", evaluate_fed_models)
    graph.add_node("write_fed_report", write_fed_report)

    graph.set_entry_point("prepare_macro_features")
    graph.add_edge("prepare_macro_features", "evaluate_fed_models")
    graph.add_edge("evaluate_fed_models", "write_fed_report")
    graph.add_edge("write_fed_report", END)
    return graph.compile()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"d:\Projects\Trade2026\modules\Ver_2026_09_01")
    from fed_data_agent import build_fed_graph
    from config import get_config

    fed_graph = build_fed_graph()
    fed = fed_graph.invoke({"config": get_config()})
    model_graph = build_fed_model_graph()
    model = model_graph.invoke({"fed_raw": fed["raw_df"], "fed_intervals": fed.get("intervals_df")})
    print(model["report"])
