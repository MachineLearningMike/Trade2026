"""
fsv_validator.py

Financial State Vector Validator

Audits FSV state quality.

Inputs:
    fsv_state.csv
    fsv_availability.csv (optional)

Outputs:
    fsv_validation_report.txt
    fsv_feature_health.csv
    fsv_outliers.csv
    fsv_correlations.csv

"""

import os
import numpy as np
import pandas as pd


STATE_FILE = "fsv_state.csv"
AVAILABILITY_FILE = "fsv_availability.csv"


REPORT_FILE = "fsv_validation_report.txt"
HEALTH_FILE = "fsv_feature_health.csv"
OUTLIER_FILE = "fsv_outliers.csv"
CORR_FILE = "fsv_correlations.csv"

VALIDATION_START_DATE = "2024-01-01"



# ==========================================================
# Load
# ==========================================================


print("Loading FSV state...")

df = pd.read_csv(
    STATE_FILE,
    parse_dates=["date"]
)

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


# ==========================================================
# Validation window
# ==========================================================

VALIDATION_START_DATE = pd.Timestamp(
    "2024-01-01"
)

df = df[
    df.date >= VALIDATION_START_DATE
].reset_index(
    drop=True
)


print(
    f"Validation range: {df.date.min()} -> {df.date.max()}"
)

print(
    f"Validation rows: {len(df)}"
)


numeric = df.select_dtypes(
    include=[np.number]
)


print(
    f"Rows: {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)



report = []

report.append(
    "FSV VALIDATION REPORT\n"
)

report.append(
    "=" * 60
)


# ==========================================================
# Basic integrity
# ==========================================================


report.append(
    "\nDATASET\n"
)

report.append(
    f"Date range: {df.date.min()} -> {df.date.max()}\n"
)

report.append(
    f"Features: {len(df.columns)-1}\n"
)



duplicates = (
    df.date
    .duplicated()
    .sum()
)


report.append(
    f"Duplicate dates: {duplicates}\n"
)



# ==========================================================
# Feature health
# ==========================================================


print(
    "Checking feature health..."
)


health = []


for col in numeric.columns:

    s = numeric[col]

    health.append(
        {
            "feature": col,

            "missing_pct":
                s.isna()
                .mean()
                *
                100,

            "unique_values":
                s.nunique(),

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


health_df = pd.DataFrame(
    health
)


health_df.to_csv(
    HEALTH_FILE,
    index=False
)



report.append(
    "\nTOP MISSING FEATURES\n"
)


report.append(
    health_df
    .sort_values(
        "missing_pct",
        ascending=False
    )
    .head(10)
    .to_string()
)



# ==========================================================
# Constant features
# ==========================================================


constant = health_df[
    health_df.unique_values <= 1
]


report.append(
    "\nCONSTANT FEATURES\n"
)

if len(constant):

    report.append(
        constant.feature
        .to_string()
    )

else:

    report.append(
        "None"
    )



# ==========================================================
# Outliers
# ==========================================================


print(
    "Detecting outliers..."
)


outliers = []


for col in numeric.columns:

    s = numeric[col]

    std = s.std()

    if (
        std is None
        or
        np.isnan(std)
        or
        std == 0
    ):
        continue


    z = (
        s - s.mean()
    ) / std


    idx = (
        z.abs()
        >
        6
    )


    for i in df[idx].index:

        outliers.append(
            {
                "date":
                    df.loc[i,"date"],

                "feature":
                    col,

                "value":
                    df.loc[i,col],

                "zscore":
                    z.loc[i],
            }
        )



outlier_df = pd.DataFrame(
    outliers
)


outlier_df.to_csv(
    OUTLIER_FILE,
    index=False
)



report.append(
    "\nOUTLIERS (|z| > 6)\n"
)

report.append(
    f"Count: {len(outlier_df)}\n"
)



# ==========================================================
# Correlation
# ==========================================================


print(
    "Checking correlations..."
)


corr = (
    numeric
    .corr()
)


pairs = []


cols = list(
    corr.columns
)


for i in range(len(cols)):

    for j in range(i+1,len(cols)):

        value = corr.iloc[i,j]

        if abs(value) >= 0.95:

            pairs.append(
                {
                    "feature1":
                        cols[i],

                    "feature2":
                        cols[j],

                    "correlation":
                        value
                }
            )


corr_df = pd.DataFrame(
    pairs
)


corr_df.to_csv(
    CORR_FILE,
    index=False
)



report.append(
    "\nHIGH CORRELATION PAIRS\n"
)

report.append(
    f"Count: {len(corr_df)}\n"
)



# ==========================================================
# Availability leakage
# ==========================================================


if os.path.exists(
    AVAILABILITY_FILE
):

    print(
        "Checking availability leakage..."
    )


    av = pd.read_csv(
        AVAILABILITY_FILE,
        parse_dates=[
            "available_time_utc",
            "observation_date"
        ]
    )


    leak = av[
        av.observation_date
        >
        av.available_time_utc
    ]


    report.append(
        "\nLEAKAGE CHECK\n"
    )


    report.append(
        f"Leakage rows: {len(leak)}\n"
    )


else:

    report.append(
        "\nAvailability file not found.\n"
    )



# ==========================================================
# Save report
# ==========================================================


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(report)
    )



print()
print(
    "="*60
)

print(
    "Validation complete."
)

print(
    "Generated:"
)

print(
    REPORT_FILE
)

print(
    HEALTH_FILE
)

print(
    OUTLIER_FILE
)

print(
    CORR_FILE
)

print(
    "="*60
)