"""
fsv_01_downloader.py

Generic Feature State Vector downloader.

Currently supports:
    - FRED
    - TwelveData

The downloader itself knows nothing about individual features.
Everything is driven by fsv_dictionary.py.

For FRED series with use_vintage_data=True, also downloads ALFRED
first-publication dates as ``{feature}__first_release`` columns so
fsv_02 can use real release timing instead of lag heuristics.
"""

import os
from datetime import datetime

import pandas as pd

from fsv_dictionary import FEATURES
from fsv_client_fred import FREDClient
from fsv_client_twelvedata import TwelveDataClient


class FSVDownloader:

    def __init__(
        self,
        fred_api_key=None,
        twelvedata_api_key=None,
    ):
        self.clients = {}

        if fred_api_key is not None:
            self.clients["FRED"] = FREDClient(fred_api_key)

        if twelvedata_api_key is not None:
            self.clients["TwelveData"] = TwelveDataClient(twelvedata_api_key)

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
            raise RuntimeError(f"No client configured for {source}")

        client = self.clients[source]
        print(f"Downloading {feature_name}")

        if source == "FRED":
            include_first = bool(meta.get("use_vintage_data", False))
            if include_first:
                print(f"  (fetching ALFRED first-release dates)")
            return client.download_series(
                feature_name=feature_name,
                series_id=meta["series"],
                start_date=start_date,
                end_date=end_date,
                include_first_release=include_first,
            )

        if source == "TwelveData":
            return client.download_symbol(
                feature_name=feature_name,
                symbol=meta["symbol"],
                interval=meta.get("interval", "1day"),
                start_date=start_date,
                end_date=end_date,
            )

        raise RuntimeError(f"Unknown source {source}")

    ####################################################################
    # Download everything
    ####################################################################

    def download_all(
        self,
        start_date="1990-01-01",
        end_date=None,
    ):
        dfs = []
        failed = []

        for feature, meta in FEATURES.items():
            try:
                df = self.download_feature(
                    feature,
                    meta,
                    start_date,
                    end_date,
                )
                # Integrity on value column
                if feature not in df.columns:
                    raise RuntimeError(f"Missing value column for {feature}")
                if df["date"].duplicated().any():
                    df = df.drop_duplicates(subset=["date"], keep="last")
                dfs.append(df)
                print(f"  ✓ {feature}: {len(df):,} rows")
            except Exception as e:
                failed.append((feature, str(e)))
                print(f"Skip {feature}: {e}")

        if len(dfs) == 0:
            raise RuntimeError("No features downloaded.")

        if failed:
            print(f"\n⚠ Skipped {len(failed)} feature(s):")
            for name, err in failed:
                print(f"  - {name}: {err}")

        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on="date", how="outer")

        merged = merged.sort_values("date").reset_index(drop=True)

        print("\nDownloaded columns:")
        print(merged.columns.tolist())

        return merged


########################################################################
# Main
########################################################################

def _resolve_api_key(env_name: str, config_key: str, config: dict, fallback: str = None):
    """Prefer env var, then config, then optional local fallback."""
    key = os.environ.get(env_name) or config.get(config_key)
    if key:
        return key
    if fallback:
        print(
            f"WARNING: {env_name} not set (and config['{config_key}'] missing). "
            f"Using local fallback key for testing."
        )
        return fallback
    return None


if __name__ == "__main__":

    from config import get_config

    config = get_config()

    # Resolution order: environment → config.py → previous local test keys
    FRED_API_KEY = _resolve_api_key(
        "FRED_API_KEY",
        "fred_api_key",
        config,
        fallback="ef542ded8a399000da02e42d5eb0321f",
    )
    TWELVEDATA_API_KEY = _resolve_api_key(
        "TWELVEDATA_API_KEY",
        "twelvedata_api_key",
        config,
        fallback="95b1a91f6be84db89dcdbbdfaf9e041c",
    )

    if not FRED_API_KEY:
        raise RuntimeError(
            "FRED API key required. Set env FRED_API_KEY or config['fred_api_key'].\n"
            "PowerShell:  $env:FRED_API_KEY = 'your_key_here'"
        )

    downloader = FSVDownloader(
        fred_api_key=FRED_API_KEY,
        twelvedata_api_key=TWELVEDATA_API_KEY,
    )

    fsv = downloader.download_all(
        start_date="2020-01-01",
    )

    print(fsv.head())

    data_dir = config.get("data_dir", "dataset_cache")
    os.makedirs(data_dir, exist_ok=True)

    fsv.to_csv(
        f"{data_dir}/fsv_raw.csv",
        index=False,
        float_format="%.8g",
    )
    print(f"Saved → {data_dir}/fsv_raw.csv")
