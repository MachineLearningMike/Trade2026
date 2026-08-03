# fsv_availability.py

"""
FSV Availability Engine

Convert raw financial observations into
information availability timestamps.

Input:
    fsv_raw_FRED.csv
    metadata.py

Output:
    fsv_availability.csv

The raw dataset is never modified.

"""

import pandas as pd
import pytz

from datetime import datetime, time

from metadata import FEATURES



# ==========================================================
# Time utilities
# ==========================================================


def localize_to_utc(
    dt,
    timezone
):
    """
    Convert local exchange/release time to UTC.
    """

    if dt is None:
        return None


    tz = pytz.timezone(
        timezone
    )


    local_dt = tz.localize(
        dt
    )


    utc_dt = local_dt.astimezone(
        pytz.UTC
    )


    return utc_dt.replace(
        tzinfo=None
    )



# ==========================================================
# Release time calculation
# ==========================================================


def compute_available_time(
    feature,
    observation_date,
):
    """
    Compute when the market could know this value.

    Returns:
        UTC timestamp
        or None if unknown
    """


    meta = FEATURES[feature]


    publish_time = meta.get(
        "publish_time"
    )


    timezone = meta.get(
        "timezone",
        "UTC"
    )


    if publish_time is None:

        return None



    hour, minute = map(
        int,
        publish_time.split(":")
    )


    release_dt = datetime.combine(
        observation_date.date(),
        time(
            hour,
            minute
        )
    )


    delay = meta.get(
        "publication_lag_days",
        0
    )


    release_dt += pd.Timedelta(
        days=delay
    )


    return localize_to_utc(
        release_dt,
        timezone
    )



# ==========================================================
# Build availability table
# ==========================================================


class FSVAvailabilityBuilder:


    def __init__(
        self,
        features
    ):

        self.features = features



    def build(
        self,
        raw_df
    ):

        rows = []


        for feature, meta in self.features.items():


            if feature not in raw_df.columns:

                continue



            print(
                f"Processing {feature:20s} "
                f"rows={raw_df[feature].notna().sum():6d}"
            )



            series = raw_df[
                [
                    "date",
                    feature
                ]
            ].dropna()



            for _, row in series.iterrows():


                obs_date = pd.Timestamp(
                    row["date"]
                )


                available_time = (
                    compute_available_time(
                        feature,
                        obs_date
                    )
                )



                rows.append(

                    {

                        "feature":
                            feature,

                        "observation_date":
                            obs_date,

                        "value":
                            row[feature],

                        "available_time_utc":
                            available_time,

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
                            ),

                    }

                )


        result = pd.DataFrame(
            rows
        )


        result = result.sort_values(
            [
                "available_time_utc",
                "feature"
            ]
        )


        return result



# ==========================================================
# Main
# ==========================================================


if __name__ == "__main__":


    RAW_FILE = (
        "fsv_raw.csv"
    )


    OUTPUT_FILE = (
        "fsv_availability.csv"
    )



    print(
        "Loading raw FSV..."
    )


    raw = pd.read_csv(
        RAW_FILE,
        parse_dates=[
            "date"
        ]
    )

    builder = FSVAvailabilityBuilder(
        FEATURES
    )


    print("\nRAW COLUMNS")
    print(raw.columns.tolist())


    print("\nMETADATA FEATURES")
    print(list(FEATURES.keys()))


    print("\nMISSING FEATURES")
    print(
        [
            f
            for f in FEATURES.keys()
            if f not in raw.columns
        ]
    )



    availability = builder.build(
        raw
    )



    print(
        availability.head(20)
    )



    print(
        "\nUnknown release times:"
    )


    print(
        availability[
            availability[
                "available_time_utc"
            ].isna()
        ]
        [
            [
                "feature",
                "frequency"
            ]
        ]
        .drop_duplicates()
    )



    availability.to_csv(
        OUTPUT_FILE,
        index=False,
        float_format="%.8g"
    )


    print(
        "\nSaved:",
        OUTPUT_FILE
    )