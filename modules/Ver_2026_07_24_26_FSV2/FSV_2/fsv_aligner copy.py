"""
fsv_aligner.py

Information-time alignment engine for Financial State Vector.

Purpose:
    Convert refined daily data into a point-in-time dataset.

Inputs:
    fsv_refined_daily.csv
    fsv_availability.csv

Output:
    fsv_aligned.csv
    fsv_alignment_report.csv

Important:
    The raw observation date is NOT used as availability.
    Only information that was released before the prediction date
    is visible.

Example:

    observation_date:
        2026-06-30

    available_time_utc:
        2026-07-24 13:00

    prediction_date:
        2026-07-10

    value:
        invisible

    prediction_date:
        2026-07-25

    value:
        visible

"""


import pandas as pd
import numpy as np



# ==========================================================
# Configuration
# ==========================================================


DAILY_FILE = "fsv_refined_daily.csv"
AVAILABILITY_FILE = "fsv_availability.csv"

OUTPUT_FILE = "fsv_aligned.csv"
REPORT_FILE = "fsv_alignment_report.csv"



# ==========================================================
# Alignment engine
# ==========================================================


class FSVAligner:


    def __init__(
        self,
        availability_df
    ):

        self.availability = (
            availability_df
            .copy()
        )


        self.availability[
            "available_time_utc"
        ] = pd.to_datetime(
            self.availability[
                "available_time_utc"
            ]
        )


        self.availability = (
            self.availability
            .sort_values(
                [
                    "feature",
                    "available_time_utc"
                ]
            )
        )



    ##################################################################
    # Align one feature
    ##################################################################


    def align_feature(
        self,
        feature,
        calendar
    ):

        df = self.availability[
            self.availability.feature == feature
        ].copy()


        if len(df) == 0:

            return pd.Series(
                np.nan,
                index=calendar,
                name=feature
            )


        df = df[
            [
                "available_time_utc",
                "value"
            ]
        ]


        df = df.sort_values(
            "available_time_utc"
        )


        left = pd.DataFrame(
            {
                "date": calendar
            }
        )


        left[
            "date"
        ] = pd.to_datetime(
            left["date"]
        )


        merged = pd.merge_asof(
            left,
            df,
            left_on="date",
            right_on="available_time_utc",
            direction="backward"
        )


        return pd.Series(
            merged["value"].values,
            index=calendar,
            name=feature
        )



    ##################################################################
    # Align all features
    ##################################################################


    def build(
        self,
        start_date=None,
        end_date=None
    ):


        features = (
            self.availability
            [
                "feature"
            ]
            .unique()
            .tolist()
        )


        if start_date is None:

            start_date = (
                self.availability
                [
                    "available_time_utc"
                ]
                .min()
                .date()
            )


        if end_date is None:

            end_date = (
                self.availability
                [
                    "available_time_utc"
                ]
                .max()
                .date()
            )



        calendar = pd.date_range(
            start_date,
            end_date,
            freq="D"
        )


        result = pd.DataFrame(
            {
                "date": calendar
            }
        )


        print(
            "\nAligning features..."
        )


        reports = []


        for feature in features:


            print(
                f"Aligning {feature}"
            )


            series = self.align_feature(
                feature,
                calendar
            )


            result[
                feature
            ] = series.values



            reports.append(

                {

                    "feature":
                        feature,

                    "observed_values":
                        series.notna().sum(),

                    "missing_values":
                        series.isna().sum(),

                    "coverage":
                        series.notna().mean(),

                }

            )


        report = pd.DataFrame(
            reports
        )


        return result, report




# ==========================================================
# Main
# ==========================================================


if __name__ == "__main__":


    print(
        "Loading availability..."
    )


    availability = pd.read_csv(
        AVAILABILITY_FILE,
        parse_dates=[
            "available_time_utc",
            "observation_date"
        ]
    )


    print(
        "Loading refined daily..."
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


    aligner = FSVAligner(
        availability
    )


    aligned, report = (
        aligner.build(
            start_date=
                daily.date.min(),
            end_date=
                daily.date.max(),
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
        float_format="%.8g"
    )


    report.to_csv(
        REPORT_FILE,
        index=False,
        float_format="%.8g"
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