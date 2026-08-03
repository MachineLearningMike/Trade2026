import pandas as pd
import requests
from datetime import datetime


class FREDCommodityDownloader:

    """
    Download Commodity State Vector from FRED.
    """


    SERIES = {

        # ============================================================
        # Energy
        # ============================================================

        "wti_oil":
            "DCOILWTICO",

        "brent_oil":
            "DCOILBRENTEU",

        "natural_gas":
            "DHHNGSP",


        # ============================================================
        # Industrial metals
        # ============================================================

        "copper":
            "PCOPPUSDM",


        # ============================================================
        # Agriculture
        # ============================================================

        "corn":
            "PMAIZMTUSDM",

        "wheat":
            "PWHEAMTUSDM",

        "soybean":
            "PSOYBUSDM",
    }


    BASE_URL = (
        "https://api.stlouisfed.org/"
        "fred/series/observations"
    )


    def __init__(
        self,
        api_key
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



    def download_csv(
        self,
        start_date="1990-01-01",
    ):


        result = None


        for name, series in self.SERIES.items():

            print(
                f"Downloading {name}"
            )


            df = self.download_series(
                series,
                start_date,
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


        result = (
            result
            .sort_values("date")
            .set_index("date")
        )


        # Daily calendar
        result = (
            result
            .reindex(
                pd.date_range(
                    result.index.min(),
                    result.index.max(),
                    freq="D"
                )
            )
            .ffill()
        )


        result.index.name = "date"


        result = result.reset_index()


        # ===============================
        # Derived features
        # ===============================


        for col in result.columns:

            if col == "date":
                continue


            result[col+"_return_1d"] = (
                result[col]
                .pct_change()
            )


            result[col+"_return_20d"] = (
                result[col]
                .pct_change(20)
            )


            result[col+"_vol_20d"] = (
                result[col+"_return_1d"]
                .rolling(20)
                .std()
            )


        # Cross commodity ratios

        if (
            "brent_oil" in result
            and
            "wti_oil" in result
        ):

            result["brent_wti"] = (
                result["brent_oil"]
                -
                result["wti_oil"]
            )


        if (
            "gold" in result
            and
            "copper" in result
        ):

            result["gold_copper_ratio"] = (
                result["gold"]
                /
                result["copper"]
            )


        return result



if __name__ == "__main__":


    API_KEY = "ef542ded8a399000da02e42d5eb0321f"


    downloader = FREDCommodityDownloader(
        API_KEY
    )


    csv = downloader.download_csv(
        start_date="1990-01-01"
    )


    print(
        csv.head()
    )


    csv.to_csv(
        "fsv_commodity_price_FRED.csv",
        index=False,
        float_format="%.8g"
    )