"""
fsv_downloader.py

Generic Feature State Vector downloader.

Currently supports:
    - FRED
    - TwelveData

The downloader itself knows nothing about individual features.
Everything is driven by feature_dictionary.py.
"""

from datetime import datetime

import pandas as pd

from fsv_dictionary import FEATURES

from fred_client import FREDClient
from twelvedata_client import TwelveDataClient


class FSVDownloader:

    def __init__(
        self,
        fred_api_key=None,
        twelvedata_api_key=None,
    ):

        self.clients = {}

        if fred_api_key is not None:
            self.clients["FRED"] = FREDClient(
                fred_api_key
            )

        if twelvedata_api_key is not None:
            self.clients["TwelveData"] = (
                TwelveDataClient(
                    twelvedata_api_key
                )
            )

    ####################################################################
    # Download one feature
    ####################################################################

    def download_feature(
        self,
        feature_name,
        meta,
        start_date="1990-01-01",
        end_date=None,
    ):

        source = meta["source"]

        if source not in self.clients:
            raise RuntimeError(
                f"No client configured for {source}"
            )

        client = self.clients[source]

        print(
            f"Downloading {feature_name}"
        )

        if source == "FRED":

            return client.download_series(
                feature_name=feature_name,
                series_id=meta["series"],
                start_date=start_date,
                end_date=end_date,
            )

        elif source == "TwelveData":

            return client.download_symbol(
                feature_name=feature_name,
                symbol=meta["symbol"],
                interval=meta.get(
                    "interval",
                    "1day",
                ),
                start_date=start_date,
                end_date=end_date,
            )

        else:

            raise RuntimeError(
                f"Unknown source {source}"
            )

    ####################################################################
    # Download everything
    ####################################################################

    def download_all(
        self,
        start_date="1990-01-01",
        end_date=None,
    ):

        dfs = []

        for feature, meta in FEATURES.items():

            try:

                df = self.download_feature(
                    feature,
                    meta,
                    start_date,
                    end_date,
                )

                dfs.append(df)

            except Exception as e:

                print(
                    f"Skip {feature}: {e}"
                )

        if len(dfs) == 0:

            raise RuntimeError(
                "No features downloaded."
            )

        merged = dfs[0]

        for df in dfs[1:]:

            merged = merged.merge(
                df,
                on="date",
                how="outer",
            )

        merged = (
            merged
            .sort_values("date")
            .reset_index(drop=True)
        )

        return merged


########################################################################
# Main
########################################################################

if __name__ == "__main__":

    FRED_API_KEY = (
        "ef542ded8a399000da02e42d5eb0321f"
    )

    TWELVEDATA_API_KEY = (
        "95b1a91f6be84db89dcdbbdfaf9e041c"
    )

    downloader = FSVDownloader(
        fred_api_key=FRED_API_KEY,
        twelvedata_api_key=TWELVEDATA_API_KEY,
    )

    fsv = downloader.download_all(
        start_date="1990-01-01",
    )

    print(fsv.head())

    fsv.to_csv(
        "fsv_raw.csv",
        index=False,
        float_format="%.8g",
    )