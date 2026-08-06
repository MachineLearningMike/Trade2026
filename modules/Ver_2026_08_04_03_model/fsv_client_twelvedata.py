"""
fsv_client_twelvedata.py

Generic TwelveData API client.

Responsibilities:
    - Download one market data series (with pagination for full history)
    - Convert timestamps
    - Convert OHLC values

No:
    - forward filling
    - resampling
    - feature engineering
    - merging
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests


class TwelveDataClient:

    BASE_URL = "https://api.twelvedata.com/time_series"
    # TwelveData free/pro plans typically cap rows per request; page with end_date.
    DEFAULT_OUTPUTSIZE = 5000

    def __init__(self, api_key: str):
        self.api_key = api_key

    ##################################################################
    # Download one symbol (paginated)
    ##################################################################

    def download_symbol(
        self,
        feature_name,
        symbol,
        interval="1day",
        start_date="2000-01-01",
        end_date=None,
        outputsize: int = DEFAULT_OUTPUTSIZE,
    ):
        """
        Download full history by walking backward with end_date pagination
        until start_date is covered or no new rows arrive.
        """
        if end_date is None:
            end_cursor = datetime.today().strftime("%Y-%m-%d")
        else:
            end_cursor = str(end_date)[:10]

        start_ts = pd.Timestamp(start_date)
        chunks = []
        seen_starts = set()
        max_pages = 200  # safety

        for _ in range(max_pages):
            if end_cursor in seen_starts:
                break
            seen_starts.add(end_cursor)

            page = self._fetch_page(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_cursor,
                outputsize=outputsize,
            )
            if page is None or page.empty:
                break

            chunks.append(page)
            earliest = pd.Timestamp(page["datetime"].min())
            if earliest <= start_ts:
                break
            # Page further back: set end just before earliest bar
            end_cursor = (earliest - timedelta(days=1)).strftime("%Y-%m-%d")
            if len(page) < outputsize:
                # API returned fewer than a full page → no more history
                break

        if not chunks:
            raise RuntimeError(f"Empty series: {symbol}")

        df = pd.concat(chunks, ignore_index=True)

        # ----------------------------------------------------------
        # Timestamp conversion
        # ----------------------------------------------------------
        if "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"])
            df = df.drop(columns=["datetime"])
        elif "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"], unit="s")
            df = df.drop(columns=["timestamp"])
        else:
            raise RuntimeError("No datetime column returned")

        # ----------------------------------------------------------
        # Numeric conversion
        # ----------------------------------------------------------
        numeric_columns = [c for c in df.columns if c != "date"]
        for c in numeric_columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # ----------------------------------------------------------
        # Rename according to feature name
        # ----------------------------------------------------------
        if "close" in df.columns:
            df = df[["date", "close"]].rename(columns={"close": feature_name})
        else:
            df = df.rename(
                columns={c: f"{feature_name}_{c}" for c in numeric_columns}
            )

        # Deduplicate overlapping pages, filter start, sort
        df = df.drop_duplicates(subset=["date"], keep="last")
        df = df[df["date"] >= start_ts]
        if end_date is not None:
            df = df[df["date"] <= pd.Timestamp(end_date)]
        df = df.sort_values("date").reset_index(drop=True)

        if df.empty:
            raise RuntimeError(f"Empty series after pagination: {symbol}")

        return df

    def _fetch_page(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
        outputsize: int,
    ) -> Optional[pd.DataFrame]:
        params = {
            "symbol": symbol,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date,
            "apikey": self.api_key,
            "format": "JSON",
            "outputsize": outputsize,
            "order": "ASC",
        }

        response = requests.get(self.BASE_URL, params=params, timeout=60)

        if response.status_code != 200:
            raise RuntimeError(
                f"TwelveData HTTP error: {response.status_code}"
            )

        data = response.json()

        if "values" not in data:
            # End of history or API message (e.g. no data for range)
            status = data.get("status") or data.get("code") or data
            if isinstance(data, dict) and data.get("code") == 400:
                return None
            # Soft-fail empty page at the back of history
            if "values" not in data:
                msg = data.get("message") or data.get("status") or str(data)
                if "No data" in str(msg) or status == "error":
                    return None
                raise RuntimeError(f"TwelveData error for {symbol}:\n{data}")

        values = data["values"]
        if len(values) == 0:
            return None

        return pd.DataFrame(values)

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
            raise ValueError("Metadata source is not TwelveData")

        df = self.download_symbol(
            feature_name=metadata.name,
            symbol=metadata.symbol,
            interval=getattr(metadata, "interval", "1day"),
            start_date=start_date,
            end_date=end_date,
        )
        self.check_integrity(df, metadata.name)
        return df

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
