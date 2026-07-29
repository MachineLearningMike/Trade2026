import requests
import pandas as pd
import time
from datetime import datetime


class InvestingGoldDownloader:

    """
    Download gold price history from Investing.com.

    Uses:
        Gold Futures
        Gold Spot

    Note:
        Investing.com endpoints are unofficial
        and may change.
    """


    HEADERS = {
        "User-Agent":
        "Mozilla/5.0"
    }


    # Investing.com instrument IDs
    # These IDs can change.
    SYMBOLS = {

        # Gold Futures (COMEX)
        "gold_future": 8830,

        # Gold Spot
        "gold_spot": 68,
    }


    def download_history(
        self,
        instrument_id,
        start_timestamp,
        end_timestamp,
    ):

        url = (
            "https://api.investing.com/api/financialdata/"
            f"historical/{instrument_id}"
        )


        params = {

            "start-date":
                start_timestamp,

            "end-date":
                end_timestamp,

            "interval":
                "P1D",

            "time-frame":
                "Daily",

            "add-missing-rows":
                "false",
        }


        r = requests.get(
            url,
            headers=self.HEADERS,
            params=params,
            timeout=30,
        )


        r.raise_for_status()


        data = r.json()


        rows = data["data"]


        df = pd.DataFrame(
            rows
        )


        return df



    def download_gold_state(
        self,
        start="2000-01-01",
    ):


        start_ts = int(
            datetime.strptime(
                start,
                "%Y-%m-%d"
            ).timestamp()
        )


        end_ts = int(
            datetime.now().timestamp()
        )


        result = {}


        for name, iid in self.SYMBOLS.items():

            print(
                f"Downloading {name}"
            )


            df = self.download_history(
                iid,
                start_ts,
                end_ts,
            )


            time.sleep(2)


            result[name] = df



        return result



if __name__ == "__main__":


    downloader = InvestingGoldDownloader()


    data = downloader.download_gold_state()


    for name, df in data.items():

        print("\n", name)

        print(
            df.head()
        )