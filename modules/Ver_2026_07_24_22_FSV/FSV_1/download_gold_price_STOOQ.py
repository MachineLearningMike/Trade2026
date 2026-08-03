import pandas as pd


class StooqGoldDownloader:


    SYMBOLS = {
        "gold_spot": "xauusd",
        "gold_future": "gc.f",
    }


    def download_one(
        self,
        symbol,
        start="20000101",
        end="20260729",
    ):

        url = (
            "https://stooq.com/q/d/l/"
            f"?s={symbol}"
            f"&d1={start}"
            f"&d2={end}"
        )


        print(
            f"Downloading {symbol}"
        )


        df = pd.read_csv(url)


        if df.empty:
            raise RuntimeError(
                f"No data for {symbol}"
            )


        df["Date"] = pd.to_datetime(
            df["Date"]
        )


        df = df[
            [
                "Date",
                "Close"
            ]
        ]


        df.columns = [
            "date",
            symbol
        ]


        return df



    def download_gold_state(self):

        spot = self.download_one(
            self.SYMBOLS["gold_spot"]
        )


        future = self.download_one(
            self.SYMBOLS["gold_future"]
        )


        df = spot.merge(
            future,
            on="date",
            how="outer"
        )


        df = (
            df
            .sort_values("date")
            .ffill()
            .dropna()
        )


        df = df.rename(
            columns={
                "xauusd": "gold_spot",
                "gc.f": "gold_future",
            }
        )


        df["gold_basis"] = (
            df["gold_future"]
            /
            df["gold_spot"]
            -
            1
        )


        return df



if __name__ == "__main__":

    downloader = StooqGoldDownloader()

    gold = downloader.download_gold_state()

    print(gold.head())

    gold.to_csv(
        "Gold_StateVector.csv",
        index=False
    )