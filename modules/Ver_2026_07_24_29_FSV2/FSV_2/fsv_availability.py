"""
fsv_availability.py

Financial State Vector Availability Engine.

Convert raw observations into
market-information availability timestamps.

Input:
    fsv_raw.csv
    metadata.py
    fsv_release_rules.py

Output:
    fsv_availability.csv

The raw dataset is never modified.

"""

import pandas as pd

from metadata import FEATURES
from fsv_release_rules import get_release_time



# ==========================================================
# Availability Builder
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

                print(
                    f"Skip missing feature: {feature}"
                )

                continue



            series = raw_df[
                [
                    "date",
                    feature
                ]
            ].dropna()



            print(
                f"Processing {feature:25s}"
                f" rows={len(series):6d}"
            )



            for _, row in series.iterrows():


                observation_date = pd.Timestamp(
                    row["date"]
                )



                available_time = get_release_time(
                    feature,
                    observation_date,
                    meta
                )



                rows.append(

                    {

                        "feature":
                            feature,


                        "observation_date":
                            observation_date,


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


        if len(result) == 0:

            raise RuntimeError(
                "No availability records created."
            )



        result = result.sort_values(
            [
                "available_time_utc",
                "feature",
            ]
        )


        result = result.reset_index(
            drop=True
        )


        return result



# ==========================================================
# Diagnostics
# ==========================================================


def report(
    raw,
    availability
):

    print(
        "\nRAW FEATURES"
    )

    print(
        raw.columns.tolist()
    )


    print(
        "\nAVAILABLE FEATURES"
    )

    print(
        availability["feature"]
        .unique()
        .tolist()
    )


    print(
        "\nUNKNOWN RELEASE TIMES"
    )


    unknown = availability[
        availability[
            "available_time_utc"
        ].isna()
    ]


    if len(unknown):

        print(

            unknown[
                [
                    "feature",
                    "frequency",
                ]
            ]
            .drop_duplicates()

        )

    else:

        print(
            "None"
        )



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



    print(
        "Building availability..."
    )


    builder = FSVAvailabilityBuilder(
        FEATURES
    )



    availability = builder.build(
        raw
    )



    print(
        "\nPreview:"
    )

    print(
        availability.head(20)
    )



    report(
        raw,
        availability
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