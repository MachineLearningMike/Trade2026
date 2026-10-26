from __future__ import annotations

from typing import Any, Dict, TypedDict

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from langgraph.graph import END, StateGraph


class ResearchState(TypedDict, total=False):
    fed_data: pd.DataFrame
    fed_summary: str
    features: pd.DataFrame
    target: pd.Series
    model_name: str
    metrics: Dict[str, float]
    status: str
    report: str


def prepare_features(state: ResearchState) -> Dict[str, Any]:
    df = state["fed_data"].copy()
    if df.empty:
        raise ValueError("No FED data available for model research.")
    df = df.sort_values("DATE").reset_index(drop=True)
    df["lag_1"] = df["value"].shift(1)
    df["lag_3"] = df["value"].shift(3)
    df["rolling_mean_3"] = df["value"].rolling(window=3, min_periods=1).mean().shift(1)
    df["target"] = df["value"].shift(-1)
    df = df.dropna().reset_index(drop=True)
    features = df[["lag_1", "lag_3", "rolling_mean_3"]]
    target = df["target"]
    state["features"] = features
    state["target"] = target
    state["status"] = "features_ready"
    return {"features": features, "target": target, "status": "features_ready"}


def evaluate_models(state: ResearchState) -> Dict[str, Any]:
    X = state["features"]
    y = state["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )

    candidates = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=42),
    }

    scores = []
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        metrics = {
            "r2": float(r2_score(y_test, predictions)),
            "rmse": float(mse ** 0.5),
            "mae": float(mean_absolute_error(y_test, predictions)),
        }
        scores.append((name, metrics))

    best_name, best_metrics = max(scores, key=lambda item: item[1]["r2"])
    state["model_name"] = best_name
    state["metrics"] = best_metrics
    state["status"] = "model_evaluated"
    return {"model_name": best_name, "metrics": best_metrics, "status": "model_evaluated"}


def write_research_report(state: ResearchState) -> Dict[str, Any]:
    fed_summary = state.get("fed_summary", "No fed summary available.")
    model_name = state.get("model_name", "unknown")
    metrics = state.get("metrics", {})
    report = (
        f"Research summary: {fed_summary}\n"
        f"Best model: {model_name}\n"
        f"Validation metrics: R2={metrics.get('r2', 0.0):.4f}, "
        f"RMSE={metrics.get('rmse', 0.0):.4f}, MAE={metrics.get('mae', 0.0):.4f}."
    )
    state["report"] = report
    state["status"] = "report_ready"
    return {"report": report, "status": "report_ready"}


def build_model_research_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("prepare_features", prepare_features)
    graph.add_node("evaluate_models", evaluate_models)
    graph.add_node("write_research_report", write_research_report)
    graph.set_entry_point("prepare_features")
    graph.add_edge("prepare_features", "evaluate_models")
    graph.add_edge("evaluate_models", "write_research_report")
    graph.add_edge("write_research_report", END)
    return graph.compile()
