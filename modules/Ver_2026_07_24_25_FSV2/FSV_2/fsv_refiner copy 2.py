"""
fsv_refiner.py

Financial State Vector Refinement Engine

Input:
    fsv_raw.csv
    metadata.py
    fsv_availability.csv

Output:
    fsv_clean.csv
    fsv_quality_report.csv

Responsibilities:

- Preserve raw data
- Validate metadata
- Align timestamps
- Handle missing values
- Apply forward-fill policy
- Remove unusable columns
- Generate quality report

No feature engineering here.

"""


import pandas as pd
import numpy as np

from metadata import FEATURES



# ==========================================================
# Configuration
# ==========================================================


RAW_FILE = "fsv_raw.csv"

AVAILABILITY_FILE = (
    "fsv_availability.csv"
)

OUTPUT_FILE = (
    "fsv_clean.csv"
)

REPORT_FILE = (
    "fsv_quality_report.csv"
)



# ==========================================================
# Quality report
# ==========================================================


def build_quality_report(df):

    rows = []


    for col in df.columns:

        if col == "date":
            continue


        s = df[col]


        rows.append(

            {

                "feature": col,

                "rows":
                    len(s),

                "available":
                    s.notna().sum(),

                "missing":
                    s.isna().sum(),

                "missing_ratio":
                    s.isna().mean(),

                "first_date":
                    df.loc[
                        s.first_valid_index(),
                        "date"
                    ]
                    if s.first_valid_index()
                    else None,

                "last_date":
                    df.loc[
                        s.last_valid_index(),
                        "date"
                    ]
                    if s.last_valid_index()
                    else None,

                "dtype":
                    str(s.dtype),

            }

        )


    return pd.DataFrame(rows)



# ==========================================================
# Frequency alignment
# ==========================================================


def align_daily(df):

    """
    Convert everything into daily observations.

    Lower frequency features remain sparse.
    They are NOT forward-filled here.
    """

    df = df.copy()


    df["date"] = pd.to_datetime(
        df["date"]
    )


    df = (
        df
        .set_index("date")
        .sort_index()
    )


    return df



# ==========================================================
# Apply forward fill rules
# ==========================================================


def apply_fill_policy(df):


    df = df.copy()


    for feature, meta in FEATURES.items():


        if feature not in df.columns:
            continue


        if meta.get(
            "forward_fill",
            False
        ):

            print(
                "Forward filling:",
                feature
            )


            df[feature] = (
                df[feature]
                .ffill()
            )


    return df



# ==========================================================
# Remove empty columns
# ==========================================================


def remove_bad_features(df):


    remove = []


    for col in df.columns:


        if (
            df[col]
            .notna()
            .sum()
            == 0
        ):

            remove.append(col)


    if remove:

        print(
            "Removing empty:",
            remove
        )

        df = df.drop(
            columns=remove
        )


    return df



# ==========================================================
# Main refinement
# ==========================================================


def refine(
    raw_file=RAW_FILE
):


    print(
        "Loading raw FSV..."
    )


    raw = pd.read_csv(
        raw_file,
        parse_dates=[
            "date"
        ]
    )


    print(
        "Raw shape:",
        raw.shape
    )



    print(
        "Checking metadata..."
    )


    unknown = [

        c for c in raw.columns

        if c != "date"
        and c not in FEATURES

    ]


    if unknown:

        print(
            "Unknown columns:",
            unknown
        )



    df = align_daily(
        raw
    )



    print(
        "Applying fill rules..."
    )


    df = apply_fill_policy(
        df
    )



    df = remove_bad_features(
        df
    )



    df = (
        df
        .reset_index()
    )



    print(
        "Clean shape:",
        df.shape
    )


    return df



# ==========================================================
# Main
# ==========================================================


if __name__ == "__main__":


    clean = refine()


    clean.to_csv(
        OUTPUT_FILE,
        index=False,
        float_format="%.8g"
    )


    report = build_quality_report(
        clean
    )


    report.to_csv(
        REPORT_FILE,
        index=False
    )


    print()
    print(
        "Saved:",
        OUTPUT_FILE
    )

    print(
        "Saved:",
        REPORT_FILE
    )


    print()
    print(report)