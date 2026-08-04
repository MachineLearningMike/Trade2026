"""
fsv_client_twelvedata.py

Generic TwelveData API client.

Responsibilities:
    - Download one market data series
    - Convert timestamps
    - Convert OHLC values

No:
    - forward filling
    - resampling
    - feature engineering
    - merging
"""


from datetime import datetime

import pandas as pd
import requests



class TwelveDataClient:


    BASE_URL = (
        "https://api.twelvedata.com/time_series"
    )


    def __init__(
        self,
        api_key: str,
    ):

        self.api_key = api_key



    ##################################################################
    # Download one symbol
    ##################################################################

    def download_symbol(
        self,
        feature_name,
        symbol,
        interval="1day",
        start_date="2000-01-01",
        end_date=None,
    ):


        params = {

            "symbol":
                symbol,

            "interval":
                interval,

            "start_date":
                start_date,

            "apikey":
                self.api_key,

            "format":
                "JSON",

        }


        if end_date is not None:

            params["end_date"] = end_date



        response = requests.get(
            self.BASE_URL,
            params=params,
            timeout=30,
        )



        if response.status_code != 200:

            raise RuntimeError(
                f"TwelveData HTTP error: "
                f"{response.status_code}"
            )



        data = response.json()



        if "values" not in data:

            raise RuntimeError(
                f"TwelveData error for "
                f"{symbol}:\n{data}"
            )



        values = data["values"]



        if len(values) == 0:

            raise RuntimeError(
                f"Empty series: {symbol}"
            )



        df = pd.DataFrame(
            values
        )



        # ----------------------------------------------------------
        # Timestamp conversion
        # ----------------------------------------------------------

        if "datetime" in df.columns:

            df["date"] = pd.to_datetime(
                df["datetime"]
            )

            df = df.drop(
                columns=[
                    "datetime"
                ]
            )


        elif "timestamp" in df.columns:

            df["date"] = pd.to_datetime(
                df["timestamp"],
                unit="s",
            )

            df = df.drop(
                columns=[
                    "timestamp"
                ]
            )


        else:

            raise RuntimeError(
                "No datetime column returned"
            )



        # ----------------------------------------------------------
        # Numeric conversion
        # ----------------------------------------------------------

        numeric_columns = [
            c for c in df.columns
            if c != "date"
        ]


        for c in numeric_columns:

            df[c] = pd.to_numeric(
                df[c],
                errors="coerce",
            )



        # ----------------------------------------------------------
        # Rename according to feature name
        #
        # For daily close:
        #     close -> feature_name
        #
        # For OHLC:
        #     keep OHLC names
        # ----------------------------------------------------------

        if "close" in df.columns:

            df = df[
                [
                    "date",
                    "close",
                ]
            ]

            df = df.rename(
                columns={
                    "close":
                        feature_name
                }
            )


        else:

            df = df.rename(
                columns={
                    c:
                        f"{feature_name}_{c}"
                    for c in numeric_columns
                }
            )



        df = (
            df
            .sort_values("date")
            .reset_index(drop=True)
        )


        return df



    ##################################################################
    # Download using FeatureMetadata
    ##################################################################

    def download_feature(
        self,
        metadata,
        start_date="2000-01-01",
        end_date=None,
    ):


        if metadata.source != "TwelveData":

            raise ValueError(
                "Metadata source is not TwelveData"
            )



        df = self.download_symbol(
            feature_name=metadata.name,

            symbol=metadata.symbol,

            interval=getattr(
                metadata,
                "interval",
                "1day",
            ),

            start_date=start_date,

            end_date=end_date,
        )


        self.check_integrity(
            df,
            metadata.name,
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