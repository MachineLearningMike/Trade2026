"""
fsv_target_builder.py

Build ML targets from Financial State Vector.

Input:
    fsv_state.csv

Output:
    fsv_training.csv
    fsv_target_report.csv

The original state file is never modified.

"""

import pandas as pd
import numpy as np


STATE_FILE = "fsv_state.csv"

OUTPUT_FILE = "fsv_training.csv"

REPORT_FILE = "fsv_target_report.csv"


# ==========================================================
# Configuration
# ==========================================================

HORIZONS = [
    1,
    5,
    20,
    60,
]


PRICE_FEATURES = {
    "sp500": "sp500",
    "vix": "vix",
    "gold": "gold_spot",
    "wti_oil": "wti_oil",
    "brent_oil": "brent_oil",
}



# ==========================================================
# Load
# ==========================================================


print(
    "Loading FSV state..."
)


df = pd.read_csv(
    STATE_FILE,
    parse_dates=[
        "date"
    ]
)


df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


print(
    "Rows:",
    len(df)
)



# ==========================================================
# Target generation
# ==========================================================


targets = pd.DataFrame(
    index=df.index
)


print(
    "Building return targets..."
)


for name, col in PRICE_FEATURES.items():

    if col not in df.columns:

        print(
            "Skip missing:",
            col
        )

        continue


    price = df[col]


    for h in HORIZONS:

        target_name = (
            f"{name}_return_{h}d"
        )


        targets[target_name] = (
            df[col]
            .shift(-h)
            /
            price
            -
            1
        )



# ==========================================================
# Volatility targets
# ==========================================================


print(
    "Building volatility targets..."
)


if "sp500" in df.columns:


    returns = (
        df["sp500"]
        .pct_change()
    )


    for h in [5,20,60]:

        targets[
            f"future_volatility_{h}d"
        ] = (
            returns
            .rolling(h)
            .std()
            .shift(-h)
        )



# ==========================================================
# State transition targets
# ==========================================================


print(
    "Building regime targets..."
)


state_columns = [

    "market_state",

    "credit_state",

    "liquidity_state",

    "rate_state",

]


for col in state_columns:

    if col in df.columns:

        targets[
            f"future_{col}_20d"
        ] = (
            df[col]
            .shift(-20)
        )



# ==========================================================
# Combine
# ==========================================================


training = pd.concat(
    [
        df,
        targets
    ],
    axis=1
)



# Remove rows where future is unknown

training = training.dropna(
    subset=[
        c
        for c in targets.columns
        if "return" in c
    ],
    how="all"
)


training = (
    training
    .reset_index(drop=True)
)



# ==========================================================
# Target report
# ==========================================================


print(
    "Creating target report..."
)


report = []


for col in targets.columns:

    s = targets[col]


    report.append(

        {

            "target":
                col,

            "count":
                s.notna().sum(),

            "missing_pct":
                s.isna()
                .mean()
                *
                100,

            "mean":
                s.mean(),

            "std":
                s.std(),

            "min":
                s.min(),

            "max":
                s.max(),

        }

    )


report_df = pd.DataFrame(
    report
)



# ==========================================================
# Save
# ==========================================================


print(
    "Saving..."
)


training.to_csv(
    OUTPUT_FILE,
    index=False,
    float_format="%.8g"
)


report_df.to_csv(
    REPORT_FILE,
    index=False,
    float_format="%.8g"
)



print()
print(
    "=" * 60
)

print(
    "Target building complete."
)

print(
    "Training:",
    OUTPUT_FILE
)

print(
    "Report:",
    REPORT_FILE
)

print(
    "Shape:",
    training.shape
)

print(
    "=" * 60
)