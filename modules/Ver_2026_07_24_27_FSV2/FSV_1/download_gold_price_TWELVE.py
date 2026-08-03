import requests
import pandas as pd


def download_gold_spot(api_key):

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": "XAU/USD",
        "interval": "1day",
        "start_date": "2000-01-01",
        "apikey": api_key,
    }

    r = requests.get(
        url,
        params=params,
        timeout=30,
    )

    data = r.json()

    if "values" not in data:
        raise RuntimeError(data)


    df = pd.DataFrame(
        data["values"]
    )


    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )


    df["close"] = pd.to_numeric(
        df["close"]
    )


    df = df.sort_values(
        "datetime"
    )


    df = df[
        [
            "datetime",
            "close"
        ]
    ]


    df.columns = [
        "date",
        "gold_spot"
    ]


    return df


if __name__ == "__main__":

    gold = download_gold_spot(
        "95b1a91f6be84db89dcdbbdfaf9e041c"
    )

    print(gold.head())

    gold.to_csv(
        "fsv_gold_price_TWELVE",
        ".csv",
        index=False
    )