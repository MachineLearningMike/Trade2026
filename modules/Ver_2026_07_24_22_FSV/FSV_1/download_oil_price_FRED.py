import pandas as pd
import requests
from datetime import datetime


class FREDOilDownloader:
    """
    Download Oil State Vector from FRED.

    Variables:
        WTI crude oil spot
        Brent crude oil spot
    """


    SERIES = {

        # ============================================================
        # Crude oil prices
        # ============================================================

        "wti_oil": "DCOILWTICO",

        "brent_oil": "DCOILBRENTEU",

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

            "series_id": series_id,

            "api_key": self.api_key,

            "file_type": "json",

            "observation_start": start_date,

            "observation_end": end_date,

        }


        r = requests.get(
            self.BASE_URL,
            params=params,
            timeout=30,
        )


        if r.status_code != 200:
            print(r.text)
            r.raise_for_status()


        data = r.json()


        observations = (
            data["observations"]
        )


        df = pd.DataFrame(
            observations
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



    def download_oil_state(
        self,
        start_date="1990-01-01",
        end_date=None,
    ):


        dfs = []


        for name, series in self.SERIES.items():

            print(
                f"Downloading {name} ({series})"
            )


            df = self.download_series(
                series,
                start_date,
                end_date,
            )


            df = df.rename(
                columns={
                    series: name
                }
            )


            check_integrity(
                df,
                name
            )


            dfs.append(df)



        # Merge
        oil = dfs[0]


        for df in dfs[1:]:

            oil = oil.merge(
                df,
                on="date",
                how="outer"
            )


        oil = (
            oil
            .sort_values("date")
            .set_index("date")
        )


        # Fill weekends/holidays
        oil = (
            oil
            .reindex(
                pd.date_range(
                    oil.index.min(),
                    oil.index.max(),
                    freq="D",
                )
            )
            .ffill()
        )


        oil.index.name = "date"


        oil = oil.reset_index()



        # ============================================================
        # Oil State Features
        # ============================================================


        oil["brent_wti_spread"] = (
            oil["brent_oil"]
            -
            oil["wti_oil"]
        )


        oil["wti_return_1d"] = (
            oil["wti_oil"]
            .pct_change()
        )


        oil["wti_return_20d"] = (
            oil["wti_oil"]
            .pct_change(20)
        )


        oil["wti_volatility_20d"] = (
            oil["wti_return_1d"]
            .rolling(20)
            .std()
        )


        oil["brent_return_1d"] = (
            oil["brent_oil"]
            .pct_change()
        )


        oil["brent_return_20d"] = (
            oil["brent_oil"]
            .pct_change(20)
        )


        oil["brent_volatility_20d"] = (
            oil["brent_return_1d"]
            .rolling(20)
            .std()
        )


        return oil



def check_integrity(
    df,
    name
):

    print(
        f"\nChecking {name}"
    )


    if df.empty:
        raise RuntimeError(
            f"{name}: empty"
        )


    if df["date"].duplicated().any():

        raise RuntimeError(
            f"{name}: duplicated dates"
        )


    missing = (
        df[name]
        .isna()
        .sum()
    )


    print(
        "Rows:",
        len(df)
    )

    print(
        "Missing:",
        missing
    )


    print(
        "Range:",
        df["date"].iloc[0].date(),
        "->",
        df["date"].iloc[-1].date()
    )


    print(
        "Integrity OK"
    )



if __name__ == "__main__":


    API_KEY = (
        "ef542ded8a399000da02e42d5eb0321f"
    )


    downloader = FREDOilDownloader(
        API_KEY
    )


    oil = downloader.download_oil_state(
        start_date="1990-01-01"
    )


    print(
        oil.head()
    )


    oil.to_csv(
        "fsv_oil_price_FRED.csv",
        index=False,
        float_format="%.8g"
    )