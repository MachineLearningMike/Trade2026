"""
fsv_state_to_utc.py

Convert fsv_state.csv date column into UTC timestamp.

Input:
    fsv_state.csv

Output:
    fsv_state_utc.csv
"""


import pandas as pd


INPUT_FILE = "fsv_state.csv"
OUTPUT_FILE = "fsv_state_utc.csv"


# ==========================================================
# Load
# ==========================================================

print("Loading FSV state...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["date"]
)


df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


print(
    "Original range:",
    df.date.min(),
    "->",
    df.date.max()
)


# ==========================================================
# Convert to UTC timestamp
# ==========================================================

print("Creating UTC timestamps...")


df["timestamp_utc"] = (
    df["date"]
    .dt.tz_localize(
        "UTC"
    )
    +
    pd.Timedelta(
        hours=23,
        minutes=59
    )
)


# Move timestamp first

cols = [
    "timestamp_utc"
] + [
    c for c in df.columns
    if c != "timestamp_utc"
]


df = df[cols]


# ==========================================================
# Save
# ==========================================================

print("Saving...")


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("="*60)

print(
    "Conversion complete"
)

print(
    "Rows:",
    len(df)
)

print(
    "Saved:",
    OUTPUT_FILE
)

print("="*60)