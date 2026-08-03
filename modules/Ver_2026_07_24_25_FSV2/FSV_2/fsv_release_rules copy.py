"""
fsv_release_rules.py

Financial State Vector release timing engine.

Purpose:
    Determine when a financial observation became available
    to the market.

Input:
    feature metadata
    observation date

Output:
    UTC availability timestamp


This module contains no data downloading logic.

"""

from datetime import datetime, time, timedelta

import pandas as pd
import pytz



# ==========================================================
# Timezone utilities
# ==========================================================


def to_utc(
    dt,
    timezone
):

    tz = pytz.timezone(
        timezone
    )

    local = tz.localize(
        dt
    )

    utc = local.astimezone(
        pytz.UTC
    )

    return utc.replace(
        tzinfo=None
    )



# ==========================================================
# Trading day utilities
# ==========================================================


def previous_business_day(
    dt
):

    while dt.weekday() >= 5:

        dt -= timedelta(
            days=1
        )

    return dt



def next_business_day(
    dt
):

    while dt.weekday() >= 5:

        dt += timedelta(
            days=1
        )

    return dt



# ==========================================================
# Release rules
# ==========================================================


def daily_market_close(
    observation_date,
    meta
):
    """
    Equity, commodity, Treasury market close.

    Typical:
        16:15 ET

    """

    d = previous_business_day(
        observation_date
    )


    release = datetime.combine(
        d,
        time(
            16,
            15
        )
    )


    return to_utc(
        release,
        meta["timezone"]
    )



def fred_daily(
    observation_date,
    meta
):
    """
    Daily FRED economic series.

    Most daily FRED series become available
    after market close.
    """

    d = next_business_day(
        observation_date
    )


    # release_time = meta.get(
    #     "publish_time",
    #     "16:00"
    # )
    release_time = meta.get(
        "publish_time"
    ) or "16:00"


    hour, minute = map(
        int,
        release_time.split(":")
    )


    release = datetime.combine(
        d,
        time(
            hour,
            minute
        )
    )


    return to_utc(
        release,
        meta["timezone"]
    )



def fred_weekly(
    observation_date,
    meta
):
    """
    Weekly FRED series.

    Examples:
        WALCL
        Mortgage rates
    """

    release = (
        observation_date
        +
        timedelta(
            days=7
        )
    )


    release = next_business_day(
        release
    )


    # release_time = meta.get(
    #     "publish_time",
    #     "16:30"
    # )
    release_time = meta.get(
        "publish_time"
    ) or "16:30"


    hour, minute = map(
        int,
        release_time.split(":")
    )


    release = datetime.combine(
        release.date(),
        time(
            hour,
            minute
        )
    )


    return to_utc(
        release,
        meta["timezone"]
    )



def fred_monthly(
    observation_date,
    meta
):
    """
    Monthly macro releases.

    Conservative approximation.

    Example:
        M2
        Housing starts

    """

    # release next month
    year = observation_date.year
    month = observation_date.month


    if month == 12:

        year += 1
        month = 1

    else:

        month += 1



    release = datetime(
        year,
        month,
        15
    )


    release_time = meta.get(
        "publish_time",
        "08:30"
    )
    release_time = meta.get(
        "publish_time"
    ) or "08:30"


    hour, minute = map(
        int,
        release_time.split(":")
    )


    release = datetime.combine(
        release.date(),
        time(
            hour,
            minute
        )
    )


    return to_utc(
        release,
        meta["timezone"]
    )



def fred_quarterly(
    observation_date,
    meta
):
    """
    Quarterly macro releases.

    Example:
        FHFA price index

    Conservative:
        ~60 days after quarter end
    """


    quarter_end = (
        pd.Timestamp(
            observation_date
        )
        .to_period("Q")
        .end_time
    )


    release = (
        quarter_end
        +
        pd.Timedelta(
            days=60
        )
    )


    release = datetime.combine(
        release.date(),
        time(
            9,
            0
        )
    )


    return to_utc(
        release,
        meta["timezone"]
    )



# ==========================================================
# Main dispatcher
# ==========================================================


def get_release_time(
    feature,
    observation_date,
    meta
):
    """
    Main interface.

    Called by fsv_availability.py
    """

    frequency = meta.get(
        "frequency",
        "daily"
    )


    source = meta.get(
        "source"
    )


    category = meta.get(
        "category"
    )


    # Market prices

    if category in [
        "equity",
        "commodity",
        "volatility",
    ]:

        return daily_market_close(
            observation_date,
            meta
        )


    # FRED daily

    if frequency == "daily":

        return fred_daily(
            observation_date,
            meta
        )


    if frequency == "weekly":

        return fred_weekly(
            observation_date,
            meta
        )


    if frequency == "monthly":

        return fred_monthly(
            observation_date,
            meta
        )


    if frequency == "quarterly":

        return fred_quarterly(
            observation_date,
            meta
        )


    return None