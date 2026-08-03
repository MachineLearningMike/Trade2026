import pandas as pd

results = pd.read_csv(
    "fsv_ablation_results.csv"
)

print(
    results
    .groupby("group")
    [
        ["rmse","r2"]
    ]
    .mean()
    .sort_values(
        "rmse"
    )
)


print("\n--- Stability ---")

stability = pd.read_csv(
    "fsv_feature_stability.csv"
)

print(
    stability.head(30)
)