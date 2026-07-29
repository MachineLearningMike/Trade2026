import pandas as pd
import requests
from datetime import datetime


class FREDRealEstateDownloader:


    SERIES = {

        # ==========================
        # Prices
        # ==========================

        "case_shiller":
            "CSUSHPINSA",

        "fhfa_price":
            "USSTHPI",


        # ==========================
        # Activity
        # ==========================

        "housing_starts":
            "HOUST",

        "building_permits":
            "PERMIT",

        "new_home_sales":
            "HSN1F",


        # ==========================
        # Mortgage
        # ==========================

        "mortgage_30y":
            "MORTGAGE30US",

    }


    BASE_URL = (
        "https://api.stlouisfed.org/"
        "fred/series/observations"
    )


    def __init__(self, api_key):

        self.api_key = api_key



    def download_series(
        self,
        series_id,
        start_date="1990-01-01"
    ):

        params = {

            "series_id":
                series_id,

            "api_key":
                self.api_key,

            "file_type":
                "json",

            "observation_start":
                start_date,

        }


        r = requests.get(
            self.BASE_URL,
            params=params,
            timeout=30
        )


        if r.status_code != 200:

            print(r.text)

            return None


        obs = r.json()["observations"]


        df = pd.DataFrame(obs)


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


        return df.rename(
            columns={
                "value": series_id
            }
        )



    def download_resv(
        self,
        start_date="1990-01-01"
    ):


        result = None


        for name, series in self.SERIES.items():

            print(
                "Downloading",
                name
            )


            df = self.download_series(
                series,
                start_date
            )


            if df is None:
                continue


            df = df.rename(
                columns={
                    series:name
                }
            )


            if result is None:

                result = df

            else:

                result = result.merge(
                    df,
                    on="date",
                    how="outer"
                )


        result = (
            result
            .sort_values("date")
        )

        # Count missing before filling

        missing_before = result.isna().sum()

        print(
            "Missing before fill:"
        )

        print(
            missing_before[
                missing_before > 0
            ]
        )


        # Forward fill

        result = result.ffill()


        # Remaining missing values

        missing_after = result.isna().sum()

        print(
            "Missing after fill:"
        )

        print(
            missing_after[
                missing_after > 0
            ]
        )


        return result



if __name__ == "__main__":


    API_KEY = "ef542ded8a399000da02e42d5eb0321f"


    downloader = FREDRealEstateDownloader(
        API_KEY
    )


    resv = downloader.download_resv(
        "1990-01-01"
    )


    print(resv.head())


    resv.to_csv(
        "fsv_real_estate_price_FRED.csv",
        index=False
    )