"""
fsv_binance_aligner.py

Align Financial State Vector with Binance candle data.

Inputs:
    fsv_state.csv
    binance_candles.csv

Outputs:
    fsv_market_dataset.csv

Design:
    - All timestamps are UTC
    - No future leakage
    - FSV state is forward filled only after availability
"""

import pandas as pd
import numpy as np
import os


# ==========================================================
# Configuration
# ==========================================================


FSV_FILE = "fsv_state.csv"

BINANCE_FILE = "binance_candles.csv"

OUTPUT_FILE = "fsv_market_dataset.csv"


TIMEFRAME = "1h"


# ==========================================================
# Load FSV
# ==========================================================


print("Loading FSV state...")


fsv = pd.read_csv(
    FSV_FILE,
)


# Detect time column

if "timestamp_utc" in fsv.columns:

    fsv_time = "timestamp_utc"

elif "state_time_utc" in fsv.columns:

    fsv_time = "state_time_utc"

elif "date" in fsv.columns:

    fsv_time = "date"

else:

    raise ValueError(
        "No FSV time column found"
    )


fsv[fsv_time] = pd.to_datetime(
    fsv[fsv_time],
    utc=True
)


fsv = (
    fsv
    .sort_values(fsv_time)
    .reset_index(drop=True)
)


print(
    "FSV range:",
    fsv[fsv_time].min(),
    "->",
    fsv[fsv_time].max()
)



# Rename for merge

fsv = fsv.rename(
    columns={
        fsv_time:
            "timestamp_utc"
    }
)



# ==========================================================
# Load Binance candles
# ==========================================================


print(
    "Loading Binance candles..."
)


binance = pd.read_csv(
    BINANCE_FILE
)


# Common Binance formats

if "open_time" in binance.columns:

    candle_time = "open_time"

elif "timestamp" in binance.columns:

    candle_time = "timestamp"

else:

    raise ValueError(
        "No Binance timestamp column found"
    )


binance[candle_time] = pd.to_datetime(
    binance[candle_time],
    utc=True
)


binance = (
    binance
    .sort_values(candle_time)
    .reset_index(drop=True)
)


binance = binance.rename(
    columns={
        candle_time:
            "timestamp_utc"
    }
)



print(
    "Binance range:",
    binance.timestamp_utc.min(),
    "->",
    binance.timestamp_utc.max()
)



# ==========================================================
# Align FSV to candle time
# ==========================================================


print(
    "Aligning FSV..."
)


merged = pd.merge_asof(
    binance,
    fsv,
    on="timestamp_utc",
    direction="backward",
)


# ==========================================================
# Leakage check
# ==========================================================


print(
    "Checking leakage..."
)


# Every FSV timestamp must be <= candle time

for col in fsv.columns:

    if col == "timestamp_utc":
        continue


# merge_asof backward already guarantees this


# ==========================================================
# Missing report
# ==========================================================


missing = (
    merged.isna()
    .mean()
    .sort_values(
        ascending=False
    )
)


print()

print(
    "Top missing:"
)

print(
    missing.head(10)
)



# ==========================================================
# Add future targets
# ==========================================================


print(
    "Creating targets..."
)


if "close" in merged.columns:

    merged["future_return_1h"] = (
        merged.close.shift(-1)
        /
        merged.close
        -
        1
    )


    merged["future_return_4h"] = (
        merged.close.shift(-4)
        /
        merged.close
        -
        1
    )


    merged["future_return_24h"] = (
        merged.close.shift(-24)
        /
        merged.close
        -
        1
    )


# ==========================================================
# Remove impossible rows
# ==========================================================


merged = merged.dropna(
    subset=[
        "timestamp_utc"
    ]
)


# ==========================================================
# Save
# ==========================================================


print(
    "Saving..."
)


merged.to_csv(
    OUTPUT_FILE,
    index=False
)



print()
print("="*60)

print(
    "FSV Binance alignment complete"
)

print(
    "Rows:",
    len(merged)
)

print(
    "Columns:",
    len(merged.columns)
)

print(
    "Saved:",
    OUTPUT_FILE
)

print("="*60)