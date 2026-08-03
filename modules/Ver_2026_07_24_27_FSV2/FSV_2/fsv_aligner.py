"""
fsv_aligner.py

Information-time based FSV aligner.

Purpose:
Convert refined raw financial data into a daily Financial State Vector.

Important:
The model only sees information that was available
at the decision timestamp.

Example:

Observation:
    2026-07-29 Treasury yield

Published:
    2026-07-29 16:15 NY time

Decision:
    2026-07-30 08:00 NY time

Available:
    YES

"""

import pandas as pd
import pytz

from datetime import datetime, time

from metadata import FEATURES
from fsv_timestamp_config import (
    PREDICTION_TIME,
    MAX_STALENESS_DAYS,
)


# ==========================================================
# Time conversion
# ==========================================================


def make_prediction_timestamp(date):
    """
    Create daily prediction timestamp.

    Example:
        2026-07-30 08:00 America/New_York

    Return:
        UTC naive timestamp
    """

    tz = pytz.timezone(
        PREDICTION_TIME["timezone"]
    )

    local_dt = tz.localize(
        datetime.combine(
            date.date(),
            time(
                PREDICTION_TIME["hour"],
                PREDICTION_TIME["minute"],
            ),
        )
    )

    utc_dt = local_dt.astimezone(
        pytz.UTC
    )

    return utc_dt.replace(
        tzinfo=None
    )



# ==========================================================
# Staleness
# ==========================================================


def get_max_staleness(meta):

    freq = meta.get(
        "frequency",
        "daily"
    )

    return MAX_STALENESS_DAYS.get(
        freq,
        30
    )



# ==========================================================
# Align one feature
# ==========================================================


def align_feature(
    availability,
    daily_dates,
    feature,
):

    meta = FEATURES[feature]

    print(
        f"Aligning {feature}"
    )


    data = availability[
        availability["feature"]
        ==
        feature
    ].copy()


    if len(data) == 0:

        return pd.Series(
            index=daily_dates,
            dtype=float,
            name=feature,
        )


    data = data.sort_values(
        "available_time_utc"
    )


    timestamps = (
        data[
            "available_time_utc"
        ]
        .values
    )


    values = (
        data[
            "value"
        ]
        .values
    )


    max_age = pd.Timedelta(
        days=get_max_staleness(meta)
    )


    result = []


    for d in daily_dates:

        prediction_time = (
            make_prediction_timestamp(
                pd.Timestamp(d)
            )
        )


        idx = (
            data[
                "available_time_utc"
            ]
            <= prediction_time
        )


        if not idx.any():

            result.append(
                None
            )

            continue


        candidate = data[
            idx
        ].iloc[-1]


        age = (
            prediction_time
            -
            candidate[
                "available_time_utc"
            ]
        )


        if age > max_age:

            result.append(
                None
            )

        else:

            result.append(
                candidate["value"]
            )


    return pd.Series(
        result,
        index=daily_dates,
        name=feature,
    )


import os


def find_daily_file():

    candidates = [

        "fsv_daily.csv",
        "fsv_refined_daily.csv",
        "fsv_daily_refined.csv",
        "fsv_refined.csv",

    ]


    for f in candidates:

        if os.path.exists(f):
            return f


    raise FileNotFoundError(
        "Cannot find refined daily FSV file.\n"
        f"Checked: {candidates}"
    )


# ==========================================================
# Main builder
# ==========================================================


class FSVAligner:


    def __init__(
        self,
        availability,
    ):

        self.availability = availability



    def build(
        self,
        daily_df,
    ):

        dates = (
            daily_df["date"]
            .sort_values()
            .unique()
        )


        output = pd.DataFrame(
            {
                "date":
                    dates
            }
        )


        report = []


        features = sorted(
            FEATURES.keys()
        )


        for feature in features:


            if feature not in set(
                self.availability["feature"]
            ):

                continue


            aligned = align_feature(
                self.availability,
                dates,
                feature,
            )


            output[feature] = (
                aligned.values
            )


            report.append(

                {
                    "feature":
                        feature,

                    "rows":
                        aligned.notna().sum(),

                    "missing":
                        aligned.isna().sum(),

                    "coverage":
                        aligned.notna().mean(),

                    "frequency":
                        FEATURES[feature]
                        .get("frequency"),

                }

            )


        report = pd.DataFrame(
            report
        )


        return output, report



# ==========================================================
# Run
# ==========================================================


if __name__ == "__main__":


    AVAILABILITY_FILE = (
        "fsv_availability.csv"
    )

    DAILY_FILE = (
        "fsv_daily.csv"
    )


    OUTPUT_FILE = (
        "fsv_aligned.csv"
    )


    REPORT_FILE = (
        "fsv_alignment_report.csv"
    )



    print(
        "Loading availability..."
    )


    availability = pd.read_csv(

        AVAILABILITY_FILE,

        parse_dates=[
            "observation_date",
            "available_time_utc",
        ]

    )



    print(
        "Loading refined daily..."
    )


    # daily = pd.read_csv(

    #     DAILY_FILE,

    #     parse_dates=[
    #         "date"
    #     ]

    # )

    DAILY_FILE = find_daily_file()

    print(
        "Using daily file:",
        DAILY_FILE
    )


    daily = pd.read_csv(
        DAILY_FILE,
        parse_dates=[
            "date"
        ]
    )


    print(
        "\nDaily range:"
    )

    print(
        daily.date.min(),
        "->",
        daily.date.max()
    )



    print(
        "\nAligning features..."
    )


    aligner = FSVAligner(
        availability
    )


    aligned, report = (
        aligner.build(
            daily
        )
    )



    print(
        "\nPreview:"
    )

    print(
        aligned.head()
    )



    print(
        "\nSaving..."
    )


    aligned.to_csv(
        OUTPUT_FILE,
        index=False,
        float_format="%.8g",
    )


    report.to_csv(
        REPORT_FILE,
        index=False,
        float_format="%.8g",
    )


    print(
        "\nSaved:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        REPORT_FILE
    )