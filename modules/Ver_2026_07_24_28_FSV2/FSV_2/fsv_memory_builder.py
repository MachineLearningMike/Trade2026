"""
fsv_memory_builder.py

Financial State Vector Minute Memory Builder

Converts sparse FSV states into 1-minute memory states.

Input:
    fsv_state_utc.csv

Output:
    fsv_state_1m_memory.csv


For each feature:

    value
    age_minutes

No:
    - decay calculation
    - interpolation
    - normalization
    - feature engineering

Decay is handled in a later phase.
"""

import pandas as pd
import numpy as np


# ==========================================================
# Files
# ==========================================================

INPUT_FILE = "fsv_state_utc.csv"

OUTPUT_FILE = "fsv_state_1m_memory.csv"


TIME_COLUMN = "date"


START_DATE = "2024-01-01"


# ==========================================================
# Load
# ==========================================================


print("Loading FSV UTC state...")


df = pd.read_csv(
    INPUT_FILE,
    parse_dates=[TIME_COLUMN]
)


df = (
    df
    .sort_values(TIME_COLUMN)
    .reset_index(drop=True)
)


df = df[
    df[TIME_COLUMN] >= START_DATE
]


print(
    f"Rows: {len(df)}"
)

print(
    f"Features: {len(df.columns)-1}"
)


# ==========================================================
# Timeline
# ==========================================================


start = df[TIME_COLUMN].min()

end = df[TIME_COLUMN].max()


timeline = pd.DataFrame(
    {
        TIME_COLUMN:
        pd.date_range(
            start=start.floor("min"),
            end=end.floor("min"),
            freq="1min",
            tz="UTC"
        )
    }
)


# remove timezone mismatch if input is naive UTC

if df[TIME_COLUMN].dt.tz is None:

    df[TIME_COLUMN] = (
        df[TIME_COLUMN]
        .dt
        .tz_localize("UTC")
    )


timeline[TIME_COLUMN] = (
    timeline[TIME_COLUMN]
    .dt
    .tz_convert(None)
)


df[TIME_COLUMN] = (
    df[TIME_COLUMN]
    .dt
    .tz_convert(None)
)



# ==========================================================
# Build memory
# ==========================================================


result = timeline.copy()


features = [
    c for c in df.columns
    if c != TIME_COLUMN
]


print(
    "Building minute memory..."
)


for feature in features:

    print(
        feature
    )


    source = df[
        [
            TIME_COLUMN,
            feature
        ]
    ].copy()


    source = source.dropna(
        subset=[feature]
    )


    if len(source) == 0:

        print(
            f"Skip empty feature: {feature}"
        )

        continue


    source = (
        source
        .sort_values(TIME_COLUMN)
    )


    # latest available observation

    merged = pd.merge_asof(
        timeline,
        source,
        on=TIME_COLUMN,
        direction="backward"
    )


    result[feature] = (
        merged[feature]
    )


    # recover availability timestamp

    availability = pd.merge_asof(
        timeline,
        source.rename(
            columns={
                TIME_COLUMN:
                "available_time"
            }
        ),
        left_on=TIME_COLUMN,
        right_on="available_time",
        direction="backward"
    )


    result[
        feature + "_age_minutes"
    ] = (
        (
            result[TIME_COLUMN]
            -
            availability["available_time"]
        )
        .dt
       .total_seconds()
        /
        60
    )



# ==========================================================
# Cleanup
# ==========================================================


result = result[
    [
        c for c in result.columns
    ]
]


result.to_csv(
    OUTPUT_FILE,
    index=False
)



print()
print("="*60)

print(
    "Memory building complete."
)

print(
    f"Rows: {len(result)}"
)

print(
    f"Columns: {len(result.columns)}"
)

print(
    OUTPUT_FILE
)

print("="*60)