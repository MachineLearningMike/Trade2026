import pandas as pd
import requests
from io import StringIO


def download_stooq_gold_future():

    url = (
        "https://stooq.com/q/d/l/"
        "?s=gc.f"
        "&d1=20000101"
        "&d2=20260729"
    )


    r = requests.get(url)

    r.raise_for_status()


    df = pd.read_csv(
        StringIO(r.text)
    )


    print(df.head())


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
        "gold_future"
    ]


    return df



if __name__ == "__main__":

    gold = download_stooq_gold_future()

    gold.to_csv(
        "gold_future.csv",
        index=False
    )