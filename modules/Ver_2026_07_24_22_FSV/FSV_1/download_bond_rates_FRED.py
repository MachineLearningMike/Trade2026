import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests


class FREDDownloader:
    """
    Download an Interest Rate State Vector (IRSV) from FRED.
    """

    SERIES = {
        # ============================================================
        # Monetary policy
        # ============================================================
        "fed_funds": "DFF",

        # ============================================================
        # Treasury yields
        # ============================================================
        "1m": "DGS1MO",
        "3m": "DGS3MO",
        "6m": "DGS6MO",
        "1y": "DGS1",
        "2y": "DGS2",
        "3y": "DGS3",
        "5y": "DGS5",
        "7y": "DGS7",
        "10y": "DGS10",
        "20y": "DGS20",
        "30y": "DGS30",

        # ============================================================
        # Real interest rates
        # ============================================================
        "tips5y": "DFII5",
        "tips10y": "DFII10",
        "tips20y": "DFII20",
        "tips30y": "DFII30",

        # ============================================================
        # Corporate bonds - Moody's long history
        # ============================================================
        "aaa_moodys": "AAA",
        "baa_moodys": "BAA",

        # ============================================================
        # Corporate bonds - ICE BofA investment grade
        # ============================================================
        "corp_aaa": "BAMLC0A1CAAA",
        "corp_aa": "BAMLC0A2CAA",
        "corp_a": "BAMLC0A3CA",
        "corp_bbb": "BAMLC0A4CBBB",
        "corp_all": "BAMLC0A0CM",

        # ============================================================
        # Corporate bonds - ICE BofA high yield
        # ============================================================
        "high_yield": "BAMLH0A0HYM2",

        # ============================================================
        # Funding markets
        # ============================================================
        "sofr": "SOFR",
    }

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def download_series(
        self,
        series_id,
        start_date="2024-01-01",
        end_date=None,
    ):
        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
        }

        r = requests.get(self.BASE_URL, params=params)
        r.raise_for_status()

        observations = r.json()["observations"]

        df = pd.DataFrame(observations)

        df = df[["date", "value"]]

        df["date"] = pd.to_datetime(df["date"])

        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        df = df.rename(columns={"value": series_id})

        return df

    def download_irsv(
        self,
        start_date="2024-01-01",
        end_date=None,
    ):

        dfs = []

        for name, series in self.SERIES.items():

            print(f"Downloading {name:12s} ({series})")

            df = self.download_series(
                series,
                start_date,
                end_date,
            )

            df = df.rename(columns={series: name})

            check_integrity(df, name)

            dfs.append(df)

        merged = dfs[0]

        for df in dfs[1:]:
            merged = merged.merge(df, on="date", how="outer")

        merged = merged.sort_values("date")

        merged = merged.reset_index(drop=True)

        merged = merged.set_index("date")

        merged = merged.reindex(
            pd.date_range(
                merged.index.min(),
                merged.index.max(),
                freq="D",
            )
        )

        merged = merged.ffill()

        merged.index.name = "date"
        merged = merged.reset_index()

        # ============================================================
        # Yield curve features
        # ============================================================

        merged["spread_10y_2y"] = (
            merged["10y"] - merged["2y"]
        )

        merged["spread_10y_3m"] = (
            merged["10y"] - merged["3m"]
        )

        merged["spread_30y_10y"] = (
            merged["30y"] - merged["10y"]
        )


        # ============================================================
        # Real-rate features
        # ============================================================

        merged["breakeven_10y"] = (
            merged["10y"] - merged["tips10y"]
        )


        # ============================================================
        # Corporate credit spreads
        # ============================================================

        merged["aaa_spread"] = (
            merged["corp_aaa"] - merged["10y"]
        )

        merged["aa_spread"] = (
            merged["corp_aa"] - merged["10y"]
        )

        merged["a_spread"] = (
            merged["corp_a"] - merged["10y"]
        )

        merged["bbb_spread"] = (
            merged["corp_bbb"] - merged["10y"]
        )

        merged["high_yield_spread"] = (
            merged["high_yield"] - merged["10y"]
        )


        # ============================================================
        # Credit risk gradient
        # ============================================================

        merged["bbb_minus_aaa"] = (
            merged["corp_bbb"] - merged["corp_aaa"]
        )

        merged["hy_minus_ig"] = (
            merged["high_yield"] - merged["corp_all"]
        )

        return merged

def check_integrity(df, name):
    """
    Check integrity of one downloaded FRED series.
    """

    print(f"\nChecking {name}...")

    # Empty dataframe
    if df.empty:
        raise RuntimeError(f"{name}: downloaded dataframe is empty.")

    # Duplicate dates
    duplicates = df["date"].duplicated().sum()
    if duplicates:
        raise RuntimeError(f"{name}: {duplicates} duplicated dates found.")

    # Ascending dates
    if not df["date"].is_monotonic_increasing:
        raise RuntimeError(f"{name}: dates are not sorted.")

    # Missing values
    missing = df[name].isna().sum()
    pct = missing / len(df) * 100

    print(f"Rows      : {len(df):,}")
    print(f"Missing   : {missing:,} ({pct:.2f}%)")

    if pct > 20:
        print("WARNING: Large percentage of missing values.")

    # Large date gaps
    gaps = df["date"].diff().dt.days

    max_gap = gaps.max()

    print(f"Largest gap : {max_gap} days")

    if max_gap > 10:
        print("WARNING: Unexpected large gap detected.")

    # First/last values
    print(
        f"Range     : "
        f"{df['date'].iloc[0].date()}  ->  {df['date'].iloc[-1].date()}"
    )

    print("Integrity OK.")


if __name__ == "__main__":

    API_KEY = "ef542ded8a399000da02e42d5eb0321f"

    downloader = FREDDownloader(API_KEY)

    irsv = downloader.download_irsv(
        start_date="1990-01-01",
    )

    print(irsv.head())

    irsv.to_csv("fsv_bond_rates_FRED.csv", index=False)