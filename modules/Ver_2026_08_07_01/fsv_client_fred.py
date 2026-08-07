"""
fsv_client_fred.py

Generic FRED API client.

Responsibilities:
    - Download one FRED series (latest values)
    - Optionally attach first-publication (ALFRED) dates for no-lookahead release timing
    - Convert dates / values to numeric

No:
    - filling missing values
    - interpolation
    - feature engineering
    - resampling
"""

from datetime import datetime
from typing import Optional

import pandas as pd
import requests


class FREDClient:

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str):
        self.api_key = api_key

    ##################################################################
    # Download one FRED series (current vintage values)
    ##################################################################

    def download_series(
        self,
        feature_name,
        series_id,
        start_date="1990-01-01",
        end_date=None,
        include_first_release: bool = False,
    ):
        """
        Download observation dates and values.

        If include_first_release=True, also attaches column
        ``{feature_name}__first_release`` = first ALFRED realtime_start
        for each observation date (when the value first became public).
        """
        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
        }

        response = requests.get(self.BASE_URL, params=params, timeout=60)

        if response.status_code != 200:
            print("FRED error:")
            print(response.text)
            raise RuntimeError(f"FRED download failed: {series_id}")

        data = response.json()
        if "observations" not in data:
            raise RuntimeError(
                f"No observations returned for {series_id}\n{data}"
            )

        observations = data["observations"]
        if len(observations) == 0:
            raise RuntimeError(f"Empty series: {series_id}")

        df = pd.DataFrame(observations)
        df = df[["date", "value"]]
        df["date"] = pd.to_datetime(df["date"], utc=False)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.rename(columns={"value": feature_name})
        df = df.sort_values("date").reset_index(drop=True)

        if include_first_release:
            first_rel = self._fetch_first_release_dates(
                series_id=series_id,
                start_date=start_date,
                end_date=end_date,
            )
            if first_rel is not None and len(first_rel) > 0:
                col = f"{feature_name}__first_release"
                first_rel = first_rel.rename(columns={"first_release": col})
                df = df.merge(first_rel, on="date", how="left")
            else:
                df[f"{feature_name}__first_release"] = pd.NaT

        return df

    def _fetch_first_release_dates(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """
        ALFRED-style query: all realtime periods, then per observation date
        take min(realtime_start) as first publication date.

        Returns DataFrame with columns: date, first_release (datetime64).
        """
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
            # Full ALFRED history for this observation window
            "realtime_start": "1776-07-04",
            "realtime_end": "9999-12-31",
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=120)
            if response.status_code != 200:
                print(
                    f"⚠ FRED vintage fetch failed for {series_id} "
                    f"(HTTP {response.status_code}); using heuristic release times."
                )
                return None

            data = response.json()
            if "observations" not in data or not data["observations"]:
                print(
                    f"⚠ FRED vintage empty for {series_id}; "
                    f"using heuristic release times."
                )
                return None

            vdf = pd.DataFrame(data["observations"])
            if "realtime_start" not in vdf.columns:
                return None

            vdf["date"] = pd.to_datetime(vdf["date"], utc=False)
            vdf["realtime_start"] = pd.to_datetime(vdf["realtime_start"], utc=False)
            # Drop non-numeric placeholder rows
            vdf["value"] = pd.to_numeric(vdf["value"], errors="coerce")
            vdf = vdf.dropna(subset=["value"])

            first = (
                vdf.groupby("date", as_index=False)["realtime_start"]
                .min()
                .rename(columns={"realtime_start": "first_release"})
            )
            return first
        except Exception as e:
            print(
                f"⚠ FRED vintage fetch error for {series_id}: {e}; "
                f"using heuristic release times."
            )
            return None

    ##################################################################
    # Integrity check
    ##################################################################

    def check_integrity(self, df, feature_name):
        if df.empty:
            raise RuntimeError(f"{feature_name}: empty dataframe")

        if df["date"].duplicated().any():
            raise RuntimeError(f"{feature_name}: duplicate dates")

        if not df["date"].is_monotonic_increasing:
            raise RuntimeError(f"{feature_name}: dates not sorted")

        missing = df[feature_name].isna().sum()
        print(f"{feature_name}: {len(df):,} rows, {missing:,} missing")
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
            raise ValueError("Metadata source is not FRED")

        include_first = bool(getattr(metadata, "use_vintage_data", False))
        df = self.download_series(
            feature_name=metadata.name,
            series_id=metadata.series,
            start_date=start_date,
            end_date=end_date,
            include_first_release=include_first,
        )
        self.check_integrity(df, metadata.name)
        return df
