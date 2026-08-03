"""
fsv_refiner.py

FSV Raw Data Refinement Engine

Input:
    fsv_raw.csv
    metadata.py

Output:
    fsv_refined_observed.csv
    fsv_refined_daily.csv
    fsv_quality_report.csv

Rules:
    - Never modify raw data
    - Never create future information
    - Never interpolate economic data
    - Forward fill only when metadata allows it

"""

import os
import pandas as pd
import numpy as np

from metadata import FEATURES



# ==========================================================
# Load
# ==========================================================


def load_raw(
    filename
):

    print(
        "Loading raw FSV..."
    )

    df = pd.read_csv(
        filename,
        parse_dates=[
            "date"
        ]
    )


    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )


    return df



# ==========================================================
# Basic cleaning
# ==========================================================


def clean_raw(
    df
):

    print(
        "Cleaning raw data..."
    )


    # remove duplicate dates

    df = (
        df
        .drop_duplicates(
            subset=["date"],
            keep="last"
        )
    )


    # sort

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )


    return df



# ==========================================================
# Feature ordering
# ==========================================================


def reorder_columns(
    df
):

    columns = [
        "date"
    ]


    for feature in FEATURES:

        if feature in df.columns:

            columns.append(
                feature
            )


    return df[
        columns
    ]



# ==========================================================
# Forward fill
# ==========================================================


def apply_forward_fill(
    df
):

    print(
        "Applying forward fill rules..."
    )


    result = df.copy()


    for feature, meta in FEATURES.items():

        if feature not in result.columns:
            continue


        if meta.get(
            "forward_fill",
            False
        ):

            print(
                "Forward fill:",
                feature
            )

            result[feature] = (
                result[feature]
                .ffill()
            )


    return result



# ==========================================================
# Daily calendar
# ==========================================================


def create_daily_calendar(
    df
):

    print(
        "Creating daily calendar..."
    )


    start = df.date.min()

    end = df.date.max()


    calendar = pd.DataFrame(
        {
            "date":
            pd.date_range(
                start,
                end,
                freq="D"
            )
        }
    )


    result = calendar.merge(
        df,
        on="date",
        how="left"
    )


    return result



# ==========================================================
# Missing statistics
# ==========================================================


def build_quality_report(
    df
):

    rows = []


    for feature, meta in FEATURES.items():


        if feature not in df.columns:

            rows.append(
                {
                    "feature":
                        feature,

                    "exists":
                        False,

                    "observations":
                        0,

                    "missing_ratio":
                        1.0,

                    "frequency":
                        meta.get(
                            "frequency"
                        ),

                    "category":
                        meta.get(
                            "category"
                        )
                }
            )

            continue



        series = df[feature]


        rows.append(
            {
                "feature":
                    feature,

                "exists":
                    True,

                "observations":
                    series.notna().sum(),

                "missing_ratio":
                    series.isna().mean(),

                "first_date":
                    df.loc[
                        series.first_valid_index(),
                        "date"
                    ]
                    if series.first_valid_index()
                    else None,

                "last_date":
                    df.loc[
                        series.last_valid_index(),
                        "date"
                    ]
                    if series.last_valid_index()
                    else None,

                "frequency":
                    meta.get(
                        "frequency"
                    ),

                "category":
                    meta.get(
                        "category"
                    ),

                "source":
                    meta.get(
                        "source"
                    )
            }
        )


    return pd.DataFrame(rows)



# ==========================================================
# Outlier detection
# ==========================================================


def detect_outliers(
    df
):

    print(
        "Detecting extreme jumps..."
    )


    rows = []


    for feature in FEATURES:

        if feature not in df.columns:
            continue


        s = df[feature]


        if s.dropna().shape[0] < 20:
            continue


        pct = (
            s
            .pct_change()
        )


        # threshold = (
        #     pct.std()
        #     *
        #     8
        # )

        std = pct.dropna().std()

        if pd.isna(std) or std == 0:
            continue

        threshold = std * 8


        mask = (
            pct.abs()
            >
            threshold
        )


        count = mask.sum()


        if count > 0:

            rows.append(
                {
                    "feature":
                        feature,

                    "outlier_count":
                        int(count),

                    "threshold":
                        float(threshold)
                }
            )


    return pd.DataFrame(rows)



# ==========================================================
# Main pipeline
# ==========================================================


def refine_fsv(
    raw_file="fsv_raw.csv"
):


    raw = load_raw(
        raw_file
    )


    raw = clean_raw(
        raw
    )


    observed = reorder_columns(
        raw
    )


    observed = apply_forward_fill(
        observed
    )


    daily = create_daily_calendar(
        observed
    )


    daily = apply_forward_fill(
        daily
    )


    print(
        "Saving refined files..."
    )


    observed.to_csv(
        "fsv_refined_observed.csv",
        index=False,
        float_format="%.8g"
    )


    daily.to_csv(
        "fsv_refined_daily.csv",
        index=False,
        float_format="%.8g"
    )


    quality = build_quality_report(
        observed
    )


    outliers = detect_outliers(
        observed
    )


    if len(outliers):

        quality = quality.merge(
            outliers,
            on="feature",
            how="left"
        )


    quality.to_csv(
        "fsv_quality_report.csv",
        index=False
    )


    print(
        "Done."
    )


    print(
        "Observed:",
        observed.shape
    )


    print(
        "Daily:",
        daily.shape
    )



# ==========================================================
# Execute
# ==========================================================


if __name__ == "__main__":

    refine_fsv(
        "fsv_raw.csv"
    )