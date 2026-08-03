import pandas as pd
import requests
from datetime import datetime


class FREDSP500Downloader:
    """
    Download S&P 500 Market State Vector from FRED.
    """

    SERIES = {
        "sp500": "SP500",
    }

    BASE_URL = (
        "https://api.stlouisfed.org/fred/series/observations"
    )


    def __init__(self, api_key: str):
        self.api_key = api_key


    def download_series(
        self,
        series_id,
        start_date="1990-01-01",
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


        r = requests.get(
            self.BASE_URL,
            params=params
        )

        r.raise_for_status()


        data = r.json()

        if "observations" not in data:
            raise RuntimeError(
                f"FRED error for {series_id}: {data}"
            )


        df = pd.DataFrame(
            data["observations"]
        )


        df = df[["date", "value"]]

        df["date"] = pd.to_datetime(
            df["date"]
        )

        df["value"] = pd.to_numeric(
            df["value"],
            errors="coerce"
        )


        df = df.rename(
            columns={
                "value": series_id
            }
        )


        return df



    def download_market_state(
        self,
        start_date="1990-01-01",
        end_date=None,
    ):

        print("Downloading S&P500...")

        df = self.download_series(
            "SP500",
            start_date,
            end_date,
        )


        df = df.rename(
            columns={
                "SP500": "sp500"
            }
        )


        # Sort
        df = (
            df
            .sort_values("date")
            .reset_index(drop=True)
        )


        # ============================================================
        # Fill weekends / holidays
        # ============================================================

        df = df.set_index("date")


        df = df.reindex(
            pd.date_range(
                df.index.min(),
                df.index.max(),
                freq="D",
            )
        )


        df["sp500"] = (
            df["sp500"]
            .ffill()
        )


        df.index.name = "date"

        df = df.reset_index()



        # ============================================================
        # Derived market state features
        # ============================================================

        # Daily return
        df["sp500_return_1d"] = (
            df["sp500"]
            .pct_change()
        )


        # 20 trading-day return
        df["sp500_return_20d"] = (
            df["sp500"]
            .pct_change(20)
        )


        # 20-day volatility
        df["sp500_volatility_20d"] = (
            df["sp500_return_1d"]
            .rolling(20)
            .std()
        )


        # Drawdown
        rolling_max = (
            df["sp500"]
            .cummax()
        )

        df["sp500_drawdown"] = (
            df["sp500"]
            /
            rolling_max
            -
            1.0
        )


        return df



if __name__ == "__main__":

    API_KEY = "ef542ded8a399000da02e42d5eb0321f"


    downloader = FREDSP500Downloader(
        API_KEY
    )


    sp500_state = downloader.download_market_state(
        start_date="1990-01-01"
    )


    print(sp500_state.head())


    sp500_state.to_csv(
        "fsv_equity_state_FRED.csv",
        index=False,
        float_format="%.6g"
    )