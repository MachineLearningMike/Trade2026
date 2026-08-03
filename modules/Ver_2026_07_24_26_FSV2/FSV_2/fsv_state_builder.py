"""
fsv_state_builder.py

Financial State Vector State Builder

Input:
    fsv_aligned.csv

Output:
    fsv_state.csv

Purpose:
    Convert raw aligned financial observations into
    economically meaningful state variables.

Raw data is preserved.
Only derived columns are appended.
"""

import numpy as np
import pandas as pd


INPUT_FILE = "fsv_aligned.csv"
OUTPUT_FILE = "fsv_state.csv"


# ==========================================================
# Helpers
# ==========================================================


def safe_pct_change(series, periods):
    return (
        series
        .pct_change(periods=periods)
    )


def rolling_zscore(
    series,
    window=252
):
    mean = (
        series
        .rolling(window, min_periods=30)
        .mean()
    )

    std = (
        series
        .rolling(window, min_periods=30)
        .std()
    )

    return (
        (series - mean)
        /
        (std + 1e-9)
    )


def rolling_percentile(
    series,
    window=252
):

    def percentile(x):

        if np.isnan(x[-1]):
            return np.nan

        return (
            np.sum(x <= x[-1])
            /
            len(x)
        )


    return (
        series
        .rolling(
            window,
            min_periods=30
        )
        .apply(
            percentile,
            raw=True
        )
    )


def add_if_possible(
    df,
    name,
    value
):
    df[name] = value


# ==========================================================
# Load
# ==========================================================


print(
    "Loading aligned FSV..."
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


raw_columns = list(
    df.columns
)


print(
    f"Raw features: {len(raw_columns)-1}"
)



# ==========================================================
# Rate State
# ==========================================================


print(
    "Building rate state..."
)


if (
    "10y_treasury" in df
    and
    "2y_treasury" in df
):

    df["yield_curve_10y_2y"] = (
        df["10y_treasury"]
        -
        df["2y_treasury"]
    )


if (
    "30y_treasury" in df
    and
    "10y_treasury" in df
):

    df["yield_curve_30y_10y"] = (
        df["30y_treasury"]
        -
        df["10y_treasury"]
    )


if "10y_treasury" in df:

    df["rate_change_5d"] = (
        df["10y_treasury"]
        -
        df["10y_treasury"].shift(5)
    )

    df["rate_change_20d"] = (
        df["10y_treasury"]
        -
        df["10y_treasury"].shift(20)
    )

    df["rate_volatility_20d"] = (
        df["10y_treasury"]
        .diff()
        .rolling(20)
        .std()
    )



# ==========================================================
# Credit State
# ==========================================================


print(
    "Building credit state..."
)


credit_columns = [
    c
    for c in [
        "hy_oas",
        "bbb_oas",
        "aaa_oas",
    ]
    if c in df
]


if len(credit_columns):

    df["credit_stress_level"] = (
        df[credit_columns]
        .mean(axis=1)
    )

    df["credit_stress_change_5d"] = (
        df["credit_stress_level"]
        -
        df["credit_stress_level"]
        .shift(5)
    )

    df["credit_stress_change_20d"] = (
        df["credit_stress_level"]
        -
        df["credit_stress_level"]
        .shift(20)
    )



# ==========================================================
# Liquidity State
# ==========================================================


print(
    "Building liquidity state..."
)


if "fed_assets" in df:

    df["fed_assets_growth_3m"] = (
        df["fed_assets"]
        .pct_change(60)
    )

    df["fed_assets_growth_12m"] = (
        df["fed_assets"]
        .pct_change(252)
    )


if "m2" in df:

    df["m2_growth_3m"] = (
        df["m2"]
        .pct_change(60)
    )

    df["m2_growth_12m"] = (
        df["m2"]
        .pct_change(252)
    )



# ==========================================================
# Market State
# ==========================================================


print(
    "Building market state..."
)


if "sp500" in df:

    df["sp500_return_1d"] = (
        df["sp500"]
        .pct_change(1)
    )

    df["sp500_return_5d"] = (
        df["sp500"]
        .pct_change(5)
    )

    df["sp500_return_20d"] = (
        df["sp500"]
        .pct_change(20)
    )


if "vix" in df:

    df["vix_change_5d"] = (
        df["vix"]
        -
        df["vix"].shift(5)
    )

    df["vix_change_20d"] = (
        df["vix"]
        -
        df["vix"].shift(20)
    )



# ==========================================================
# Commodity State
# ==========================================================


print(
    "Building commodity state..."
)


if "wti_oil" in df:

    df["oil_return_5d"] = (
        df["wti_oil"]
        .pct_change(5)
    )

    df["oil_return_20d"] = (
        df["wti_oil"]
        .pct_change(20)
    )


if "gold_spot" in df:

    df["gold_return_20d"] = (
        df["gold_spot"]
        .pct_change(20)
    )



# ==========================================================
# Historical Regimes
# ==========================================================


print(
    "Building regime state..."
)


if "vix" in df:

    df["vix_zscore_252d"] = (
        rolling_zscore(
            df["vix"]
        )
    )

    df["vix_percentile_252d"] = (
        rolling_percentile(
            df["vix"]
        )
    )


if "credit_stress_level" in df:

    df["credit_zscore_252d"] = (
        rolling_zscore(
            df["credit_stress_level"]
        )
    )


if "10y_treasury" in df:

    df["rate_zscore_252d"] = (
        rolling_zscore(
            df["10y_treasury"]
        )
    )



# ==========================================================
# Composite Scores
# ==========================================================


print(
    "Building composite scores..."
)


stress_parts = []


for c in [
    "vix_zscore_252d",
    "credit_zscore_252d",
]:

    if c in df:
        stress_parts.append(
            df[c]
        )


if "sp500_return_20d" in df:

    stress_parts.append(
        -rolling_zscore(
            df["sp500_return_20d"]
        )
    )


if len(stress_parts):

    df["market_stress_score"] = (
        pd.concat(
            stress_parts,
            axis=1
        )
        .mean(axis=1)
    )



if "fed_assets_growth_12m" in df:

    df["liquidity_score"] = (
        rolling_zscore(
            df["fed_assets_growth_12m"]
        )
    )


if "credit_stress_level" in df:

    df["credit_score"] = (
        rolling_zscore(
            df["credit_stress_level"]
        )
    )



# ==========================================================
# Save
# ==========================================================


df.to_csv(
    OUTPUT_FILE,
    index=False,
    float_format="%.8g"
)


derived = [
    c
    for c in df.columns
    if c not in raw_columns
]


print()
print(
    "=============================="
)

print(
    f"Raw columns: {len(raw_columns)}"
)

print(
    f"Derived columns: {len(derived)}"
)

print(
    f"Final columns: {len(df.columns)}"
)

print(
    "Saved:",
    OUTPUT_FILE
)

print(
    "=============================="
)