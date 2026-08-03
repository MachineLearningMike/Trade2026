"""
fred_client.py

Generic FRED API client.

Responsibilities:
    - Download one FRED series
    - Convert dates
    - Convert values to numeric

No:
    - filling missing values
    - interpolation
    - feature engineering
    - resampling
"""


from datetime import datetime

import pandas as pd
import requests



class FREDClient:


    BASE_URL = (
        "https://api.stlouisfed.org/"
        "fred/series/observations"
    )


    def __init__(
        self,
        api_key: str,
    ):

        self.api_key = api_key



    ##################################################################
    # Download one FRED series
    ##################################################################

    def download_series(
        self,
        feature_name,
        series_id,
        start_date="1990-01-01",
        end_date=None,
    ):


        if end_date is None:

            end_date = (
                datetime.today()
                .strftime("%Y-%m-%d")
            )



        params = {

            "series_id":
                series_id,

            "api_key":
                self.api_key,

            "file_type":
                "json",

            "observation_start":
                start_date,

            "observation_end":
                end_date,

        }



        response = requests.get(
            self.BASE_URL,
            params=params,
            timeout=30,
        )



        if response.status_code != 200:

            print(
                "FRED error:"
            )

            print(
                response.text
            )

            raise RuntimeError(
                f"FRED download failed: "
                f"{series_id}"
            )



        data = response.json()



        if "observations" not in data:

            raise RuntimeError(
                f"No observations returned "
                f"for {series_id}\n"
                f"{data}"
            )



        observations = (
            data["observations"]
        )



        if len(observations) == 0:

            raise RuntimeError(
                f"Empty series: {series_id}"
            )



        df = pd.DataFrame(
            observations
        )



        # ----------------------------------------------------------
        # Keep only raw observation fields
        # ----------------------------------------------------------

        df = df[
            [
                "date",
                "value",
            ]
        ]



        df["date"] = pd.to_datetime(
            df["date"],
            utc=False,
        )



        df["value"] = pd.to_numeric(
            df["value"],
            errors="coerce",
        )



        df = df.rename(
            columns={
                "value":
                    feature_name
            }
        )



        df = (
            df
            .sort_values("date")
            .reset_index(drop=True)
        )



        return df



    ##################################################################
    # Integrity check
    ##################################################################

    def check_integrity(
        self,
        df,
        feature_name,
    ):


        if df.empty:

            raise RuntimeError(
                f"{feature_name}: empty dataframe"
            )



        if df["date"].duplicated().any():

            raise RuntimeError(
                f"{feature_name}: duplicate dates"
            )



        if not df["date"].is_monotonic_increasing:

            raise RuntimeError(
                f"{feature_name}: "
                "dates not sorted"
            )



        missing = (
            df[feature_name]
            .isna()
            .sum()
        )


        print(
            f"{feature_name}: "
            f"{len(df):,} rows, "
            f"{missing:,} missing"
        )


        return True



    ##################################################################
    # Download using metadata object
    ##################################################################

    def download_feature(
        self,
        metadata,
        start_date="1990-01-01",
        end_date=None,
    ):


        if metadata.source != "FRED":

            raise ValueError(
                "Metadata source is not FRED"
            )



        df = self.download_series(
            feature_name=metadata.name,

            series_id=metadata.series,

            start_date=start_date,

            end_date=end_date,
        )



        self.check_integrity(
            df,
            metadata.name,
        )



        return df