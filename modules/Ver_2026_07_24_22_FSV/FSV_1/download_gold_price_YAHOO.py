import yfinance as yf
import pandas as pd
import time


class GoldMarketDownloader:
    """
    Download gold spot and futures prices.

    Spot:
        XAUUSD=X

    Futures:
        GC=F
    """


    TICKERS = {
        "gold_spot": "XAUUSD=X",
        "gold_future": "GC=F",
    }


    def download_one(
        self,
        ticker,
        start_date="1990-01-01",
    ):

        for attempt in range(3):

            try:

                print(
                    f"Downloading {ticker}"
                )


                df = yf.download(
                    ticker,
                    start=start_date,
                    progress=False,
                    auto_adjust=False,
                    timeout=60,
                )


                if not df.empty:
                    break


            except Exception as e:

                print(
                    f"Attempt {attempt+1}: {e}"
                )

                time.sleep(5)


        if df.empty:
            raise RuntimeError(
                f"Cannot download {ticker}"
            )


        # New yfinance returns MultiIndex
        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df = df["Close"]

            df.columns = [
                ticker
            ]

        else:

            df = df[
                ["Close"]
            ]

            df.columns = [
                ticker
            ]


        df.index.name = "date"

        return df



    def download_gold_state(
        self,
        start_date="2000-01-01",
    ):


        spot = self.download_one(
            self.TICKERS["gold_spot"],
            start_date,
        )


        future = self.download_one(
            self.TICKERS["gold_future"],
            start_date,
        )


        df = pd.concat(
            [
                spot,
                future,
            ],
            axis=1
        )


        df = df.rename(
            columns={
                "XAUUSD=X": "gold_spot",
                "GC=F": "gold_future",
            }
        )


        # Fill missing holidays
        df = (
            df
            .sort_index()
            .ffill()
        )


        # Remove rows without initial data
        df = df.dropna()


        # ==================================================
        # Gold basis
        # ==================================================

        df["gold_basis"] = (
            df["gold_future"]
            /
            df["gold_spot"]
            -
            1
        )


        # ==================================================
        # Returns
        # ==================================================

        for col in [
            "gold_spot",
            "gold_future",
        ]:

            df[f"{col}_return_1d"] = (
                df[col]
                .pct_change()
            )


            df[f"{col}_return_20d"] = (
                df[col]
                .pct_change(20)
            )


            df[f"{col}_volatility_20d"] = (
                df[f"{col}_return_1d"]
                .rolling(20)
                .std()
            )


        return df.reset_index()



if __name__ == "__main__":


    downloader = GoldMarketDownloader()


    gold = downloader.download_gold_state(
        start_date="2000-01-01"
    )


    print(gold.head())


    gold.to_csv(
        "Gold_Market_StateVector.csv",
        index=False,
        float_format="%.8g"
    )