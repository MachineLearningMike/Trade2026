import pandas as pd
import requests
from datetime import datetime


class FREDGoldDownloader:
    """
    Download Gold State Vector from FRED.
    """


    SERIES = {
        "gold": "GOLDAMGBD228NLBM",
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
            end_date = datetime.today().strftime(
                "%Y-%m-%d"
            )


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

        if r.status_code != 200:
            print("FRED RESPONSE:")
            print(r.text)

            raise RuntimeError(
                f"Failed downloading {series_id}"
            )

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



    def download_gold_state(
        self,
        start_date="1990-01-01",
        end_date=None,
    ):

        print("Downloading Gold...")


        df = self.download_series(
            "GOLDPMGBD228NLBM",
            start_date,
            end_date,
        )


        df = df.rename(
            columns={
                "GOLDPMGBD228NLBM": "gold"
            }
        )


        df = (
            df
            .sort_values("date")
            .reset_index(drop=True)
        )


        # ============================================================
        # Fill weekends and holidays
        # ============================================================

        df = df.set_index("date")


        df = df.reindex(
            pd.date_range(
                df.index.min(),
                df.index.max(),
                freq="D",
            )
        )


        df["gold"] = (
            df["gold"]
            .ffill()
        )


        df.index.name = "date"

        df = df.reset_index()



        # ============================================================
        # Gold State Features
        # ============================================================


        # Daily return
        df["gold_return_1d"] = (
            df["gold"]
            .pct_change()
        )


        # Approximately one month
        df["gold_return_20d"] = (
            df["gold"]
            .pct_change(20)
        )


        # One-month volatility
        df["gold_volatility_20d"] = (
            df["gold_return_1d"]
            .rolling(20)
            .std()
        )


        # Drawdown from historical peak
        rolling_max = (
            df["gold"]
            .cummax()
        )


        df["gold_drawdown"] = (
            df["gold"]
            /
            rolling_max
            -
            1.0
        )


        return df

def search_fred_series(api_key, search_text):

    url = "https://api.stlouisfed.org/fred/series/search"

    params = {
        "search_text": search_text,
        "api_key": api_key,
        "file_type": "json",
    }

    r = requests.get(
        url,
        params=params
    )

    r.raise_for_status()

    data = r.json()

    for s in data["seriess"][:20]:
        print(
            s["id"],
            " | ",
            s["title"]
        )

def check_gold_integrity(df):

    print("\nGold State Vector Integrity")

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Range: "
        f"{df['date'].iloc[0].date()} -> "
        f"{df['date'].iloc[-1].date()}"
    )


    print("\nMissing:")

    print(
        df.isna().sum()
    )



if __name__ == "__main__":


    API_KEY = "ef542ded8a399000da02e42d5eb0321f"


    downloader = FREDGoldDownloader(
        API_KEY
    )

    search_fred_series(
        API_KEY,
        "gold price"
    )

    gold_state = downloader.download_gold_state(
        start_date="1990-01-01"
    )


    check_gold_integrity(
        gold_state
    )


    print(
        gold_state.head()
    )


    gold_state.to_csv(
        "FRED_gold_price.csv",
        index=False,
        float_format="%.6g"
    )