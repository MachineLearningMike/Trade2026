"""
fsv_ablation_test.py

Financial State Vector Ablation Test

Goal:
Measure predictive contribution of FSV feature groups.

Input:
    fsv_training.csv

Outputs:
    fsv_ablation_results.csv
    fsv_feature_stability.csv
    fsv_feature_importance.csv

"""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ==========================================================
# Configuration
# ==========================================================


INPUT_FILE = "fsv_training.csv"

RESULT_FILE = "fsv_ablation_results.csv"
STABILITY_FILE = "fsv_feature_stability.csv"
IMPORTANCE_FILE = "fsv_feature_importance.csv"


START_DATE = "2014-01-01"


TARGET = "target_return_20d"


SEEDS = [
    1,
    7,
    42,
    77,
    123
]


TEST_RATIO = 0.2



# ==========================================================
# Feature groups
# ==========================================================


FEATURE_GROUPS = {


    "market":
    [
        "sp500",
        "vix",
        "gold_spot",
        "wti_oil",
        "brent_oil",
        "dollar_index"
    ],


    "rates":
    [
        "fed_funds",
        "1y_treasury",
        "2y_treasury",
        "5y_treasury",
        "10y_treasury",
        "30y_treasury",
        "tips10y",
        "sofr"
    ],


    "credit":
    [
        "aaa_oas",
        "aa_oas",
        "a_oas",
        "bbb_oas",
        "hy_oas",
        "ted_spread",
        "commercial_paper_3m"
    ],


    "liquidity":
    [
        "fed_assets",
        "bank_reserves",
        "m2"
    ],


    "commodity":
    [
        "gold_spot",
        "wti_oil",
        "brent_oil"
    ],


    "regime":
    [
        c
        for c in []
    ]

}



# ==========================================================
# Load
# ==========================================================


print(
    "Loading training data..."
)


df = pd.read_csv(
    INPUT_FILE,
    parse_dates=[
        "date"
    ]
)


df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


df = df[
    df.date >= START_DATE
]


print(
    "Rows:",
    len(df)
)


print(
    "Columns:",
    len(df.columns)
)



# ==========================================================
# Detect target
# ==========================================================


if TARGET not in df.columns:

    candidates = [
        c
        for c in df.columns
        if "return" in c
    ]

    if len(candidates) == 0:
        raise ValueError(
            "No return target found"
        )

    TARGET = candidates[0]


print(
    "Target:",
    TARGET
)

before = len(df)

df = (
    df[
        df[TARGET].notna()
    ]
    .reset_index(drop=True)
)


after = len(df)


print(
    f"Removed rows with missing target: {before-after}"
)

# ==========================================================
# Feature detection
# ==========================================================


drop_columns = [
    "date",
    TARGET
]


all_features = [
    c
    for c in df.columns
    if c not in drop_columns
]


numeric_features = [
    c
    for c in all_features
    if pd.api.types.is_numeric_dtype(
        df[c]
    )
]


print(
    "Numeric features:",
    len(numeric_features)
)



# remove missing names

for k,v in FEATURE_GROUPS.items():

    FEATURE_GROUPS[k] = [
        x
        for x in v
        if x in numeric_features
    ]



FEATURE_GROUPS["all"] = numeric_features



# ==========================================================
# Train / test split
# ==========================================================


def split_random(
    X,
    y,
    seed
):

    rng = np.random.RandomState(
        seed
    )


    idx = np.arange(
        len(X)
    )


    rng.shuffle(
        idx
    )


    n_test = int(
        len(idx)
        *
        TEST_RATIO
    )


    test_idx = idx[:n_test]

    train_idx = idx[n_test:]


    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx]
    )



# ==========================================================
# Model
# ==========================================================


def create_model(seed):

    return Pipeline(

        [

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),


            (
                "model",
                GradientBoostingRegressor(

                    n_estimators=200,

                    learning_rate=0.03,

                    max_depth=3,

                    random_state=seed

                )
            )

        ]

    )



# ==========================================================
# Run ablation
# ==========================================================


results = []

importance_records = []



for seed in SEEDS:


    print()
    print(
        "Running seed:",
        seed
    )


    y = df[TARGET]


    for group, features in FEATURE_GROUPS.items():


        if len(features) == 0:

            print(
                "  Skip empty group:",
                group
            )

            continue


        print(
            "  ",
            group
        )


        X = df[
            features
        ]


        X_train, X_test, y_train, y_test = split_random(
            X,
            y,
            seed
        )


        model = create_model(
            seed
        )


        model.fit(
            X_train,
            y_train
        )


        pred = model.predict(
            X_test
        )


        mse = mean_squared_error(
            y_test,
            pred
        )


        r2 = r2_score(
            y_test,
            pred
        )


        results.append(

            {

                "seed":
                    seed,

                "group":
                    group,

                "features":
                    len(features),

                "mse":
                    mse,

                "rmse":
                    np.sqrt(mse),

                "r2":
                    r2

            }

        )


        if group == "all":


            imp = (
                model
                .named_steps["model"]
                .feature_importances_
            )


            for f, v in zip(
                features,
                imp
            ):

                importance_records.append(

                    {

                        "seed":
                            seed,

                        "feature":
                            f,

                        "importance":
                            v

                    }

                )



# ==========================================================
# Save
# ==========================================================


result_df = pd.DataFrame(
    results
)


result_df.to_csv(
    RESULT_FILE,
    index=False
)



imp_df = pd.DataFrame(
    importance_records
)


imp_df.to_csv(
    IMPORTANCE_FILE,
    index=False
)



stability = (

    imp_df
    .groupby(
        "feature"
    )
    .importance
    .agg(
        [
            "mean",
            "std",
            "count"
        ]
    )
    .sort_values(
        "mean",
        ascending=False
    )

)


stability.to_csv(
    STABILITY_FILE
)



print()
print("="*60)

print(
    "Ablation complete"
)

print(
    RESULT_FILE
)

print(
    STABILITY_FILE
)

print(
    IMPORTANCE_FILE
)

print("="*60)