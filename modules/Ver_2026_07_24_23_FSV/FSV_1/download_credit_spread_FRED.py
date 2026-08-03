import pandas as pd
import requests
from datetime import datetime


class FREDCreditSpreadDownloader:
    """
    Download Credit Spread State Vector (CRSV) from FRED.
    """


    SERIES = {

        # ============================================================
        # Investment grade corporate spreads (ICE BofA OAS)
        # ============================================================

        "aaa_oas":
            "BAMLC0A1CAAA",

        "aa_oas":
            "BAMLC0A2CAA",

        "a_oas":
            "BAMLC0A3CA",

        "bbb_oas":
            "BAMLC0A4CBBB",


        # ============================================================
        # High yield
        # ============================================================

        "hy_oas":
            "BAMLH0A0HYM2",

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

            print(
                r.text
            )

            return None



        data = r.json()


        if "observations" not in data:

            print(
                data
            )

            return None



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



    def download_crsv(
        self,
        start_date="1990-01-01",
        end_date=None,
    ):


        result = None


        for name, series in self.SERIES.items():

            print(
                f"Downloading {name} ({series})"
            )


            df = self.download_series(
                series,
                start_date,
                end_date,
            )


            if df is None:

                print(
                    f"Skip {name}"
                )

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



        # ============================================================
        # Daily calendar + forward fill
        # ============================================================

        result = (
            result
            .sort_values("date")
            .set_index("date")
        )


        result = result.reindex(
            pd.date_range(
                result.index.min(),
                result.index.max(),
                freq="D",
            )
        )


        result = result.ffill()


        result.index.name = "date"


        result = result.reset_index()



        # ============================================================
        # Derived credit features
        # ============================================================


        # Credit quality gradient

        result["bbb_minus_aaa"] = (
            result["bbb_oas"]
            -
            result["aaa_oas"]
        )


        # High yield risk premium

        result["hy_minus_bbb"] = (
            result["hy_oas"]
            -
            result["bbb_oas"]
        )


        # Spread momentum

        for col in [
            "aaa_oas",
            "bbb_oas",
            "hy_oas",
        ]:

            result[col + "_change_20d"] = (
                result[col]
                .diff(20)
            )


        # Spread volatility

        result["hy_volatility_20d"] = (
            result["hy_oas"]
            .diff()
            .rolling(20)
            .std()
        )


        return result



if __name__ == "__main__":


    API_KEY = (
        "ef542ded8a399000da02e42d5eb0321f"
    )


    downloader = FREDCreditSpreadDownloader(
        API_KEY
    )


    crsv = downloader.download_crsv(
        start_date="1990-01-01"
    )


    print(
        crsv.head()
    )


    crsv.to_csv(
        "fsv_credit_spread_FRED.csv",
        index=False,
        float_format="%.8g"
    )