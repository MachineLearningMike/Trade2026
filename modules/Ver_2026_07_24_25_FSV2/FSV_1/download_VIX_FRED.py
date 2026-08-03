import pandas as pd
import requests
from datetime import datetime


class FREDVIXDownloader:
    """
    Download VIX State Vector from FRED.
    """


    SERIES = {

        # CBOE Volatility Index
        "vix":
            "VIXCLS",

    }


    BASE_URL = (
        "https://api.stlouisfed.org/"
        "fred/series/observations"
    )


    def __init__(
        self,
        api_key: str
    ):
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


        r = requests.get(
            self.BASE_URL,
            params=params,
            timeout=30,
        )


        if r.status_code != 200:

            print(r.text)

            raise RuntimeError(
                "FRED download failed"
            )


        data = r.json()


        if "observations" not in data:

            print(data)

            raise RuntimeError(
                "No observations returned"
            )


        df = pd.DataFrame(
            data["observations"]
        )


        df = df[
            [
                "date",
                "value"
            ]
        ]


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



    def download_vix_state(
        self,
        start_date="1990-01-01",
        end_date=None,
    ):


        print("Downloading VIX...")


        df = self.download_series(
            "VIXCLS",
            start_date,
            end_date,
        )


        df = df.rename(
            columns={
                "VIXCLS": "vix"
            }
        )


        # ============================================================
        # Fill missing dates
        # ============================================================

        df = (
            df
            .sort_values("date")
            .set_index("date")
        )


        df = df.reindex(
            pd.date_range(
                df.index.min(),
                df.index.max(),
                freq="D",
            )
        )


        # weekends/holidays use last market close

        df = df.ffill()


        df.index.name = "date"

        df = df.reset_index()



        # ============================================================
        # Derived features
        # ============================================================


        # Absolute change

        df["vix_change_1d"] = (
            df["vix"]
            .diff(1)
        )


        # Momentum

        df["vix_change_5d"] = (
            df["vix"]
            .diff(5)
        )


        df["vix_change_20d"] = (
            df["vix"]
            .diff(20)
        )


        # Realized VIX movement

        df["vix_volatility_20d"] = (
            df["vix"]
            .diff()
            .rolling(20)
            .std()
        )


        # Historical percentile

        df["vix_percentile_252d"] = (
            df["vix"]
            .rolling(252)
            .rank(pct=True)
        )


        return df



if __name__ == "__main__":


    API_KEY = (
        "YOUR_FRED_API_KEY"
    )


    downloader = FREDVIXDownloader(
        "ef542ded8a399000da02e42d5eb0321f"
    )


    vix_state = downloader.download_vix_state(
        start_date="1990-01-01"
    )


    print(
        vix_state.head()
    )


    vix_state.to_csv(
        "fsv_VID_FRED.csv",
        index=False,
        float_format="%.8g"
    )