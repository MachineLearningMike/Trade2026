import pandas as pd
import requests
from datetime import datetime


class FREDGLSVDownloader:
    """
    Download Global Liquidity State Vector (GLSV) from FRED.
    """

    SERIES = {

        # ============================================================
        # Central bank liquidity
        # ============================================================

        "fed_assets": "WALCL",
        "bank_reserves": "TOTRESNS",


        # ============================================================
        # Money supply
        # ============================================================

        "m2": "M2SL",


        # ============================================================
        # Dollar conditions
        # ============================================================

        "dollar_index": "DTWEXBGS",


        # ============================================================
        # Funding market
        # ============================================================

        "sofr": "SOFR",
        "ted_spread": "TEDRATE",
        "commercial_paper_3m": "CPF3M",


        # ============================================================
        # Credit liquidity
        # ============================================================

        "bbb_yield": "BAMLC0A4CBBB",
        "high_yield": "BAMLH0A0HYM2",


        # ============================================================
        # Market liquidity / risk
        # ============================================================

        "vix": "VIXCLS",
        "sp500": "SP500",
    }


    BASE_URL = (
        "https://api.stlouisfed.org/fred/series/observations"
    )


    def __init__(self, api_key):
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


        obs = r.json()["observations"]


        df = pd.DataFrame(obs)

        df = df[["date", "value"]]

        df["date"] = pd.to_datetime(df["date"])

        df["value"] = pd.to_numeric(
            df["value"],
            errors="coerce"
        )

        df = df.rename(
            columns={"value": series_id}
        )


        return df


    def download_glsv(
        self,
        start_date="1990-01-01",
        end_date=None,
    ):

        dfs = []


        for name, series in self.SERIES.items():

            print(
                f"Downloading {name:20s} ({series})"
            )


            df = self.download_series(
                series,
                start_date,
                end_date
            )


            df = df.rename(
                columns={
                    series: name
                }
            )


            dfs.append(df)



        # Merge
        merged = dfs[0]


        for df in dfs[1:]:

            merged = merged.merge(
                df,
                on="date",
                how="outer"
            )


        merged = (
            merged
            .sort_values("date")
            .reset_index(drop=True)
        )


        # ============================================================
        # Derived liquidity features
        # ============================================================


        # Corporate credit stress
        merged["bbb_spread"] = (
            merged["bbb_yield"]
            -
            merged["fed_assets"] * 0
        )


        # High yield spread requires Treasury benchmark
        # Usually subtract 10Y Treasury externally
        # merged["hy_spread"] = high_yield - treasury_10y


        # Normalize market stress indicators
        merged["credit_risk_ratio"] = (
            merged["high_yield"]
            -
            merged["bbb_yield"]
        )


        return merged



def check_glsv(df):

    print("\nGLSV Integrity Check")

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Range: {df.date.min()} -> {df.date.max()}"
    )


    missing = df.isna().sum()

    print("\nMissing values:")

    print(
        missing[missing > 0]
    )



if __name__ == "__main__":

    API_KEY = "YOUR_FRED_API_KEY"


    downloader = FREDGLSVDownloader(
        "ef542ded8a399000da02e42d5eb0321f"
    )


    glsv = downloader.download_glsv(
        start_date="1990-01-01"
    )


    check_glsv(glsv)


    glsv.to_csv(
        "FRED_global_liquidity.csv",
        index=False,
        float_format="%.6g"
    )


    print(glsv.head())