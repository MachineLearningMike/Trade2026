"""
macro_02_find_value_intervals.py

Financial State Vector Availability Engine.

Input:
    macro_raw.csv
    macro_dictionary.py

Output:
    macro_value_intervals.csv

Converts each raw observation to:
    (feature, release_datetime, next_release_datetime, value)

Interval is half-open: value is valid for
    release_datetime <= t < next_release_datetime

The last observation uses an open-ended next_release (far-future sentinel)
so it remains valid after its release.

Release timing:
    1. Prefer ALFRED first-publication column ``{feature}__first_release``
       when present (from macro_01 when use_vintage_data=True).
    2. Else heuristic: observation date + release_lag_days + release_time,
       with business-day adjustment only for non-continuous markets.

The raw dataset is never modified.
"""

import time
import logging
from datetime import datetime, time as dt_time, timedelta
from typing import Optional, Dict, Any

import pandas as pd
import pytz
from pandas.tseries.holiday import USFederalHolidayCalendar

from macro_dictionary import get_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ==========================================================
# Constants
# ==========================================================

_US_HOLIDAYS = USFederalHolidayCalendar().holidays(
    start="1990-01-01",
    end="2030-12-31",
)
# Compare by calendar date (DatetimeIndex membership is unreliable with date())
_US_HOLIDAY_DATES = set(pd.Timestamp(h).date() for h in _US_HOLIDAYS)

# Open-ended upper bound for the last interval (pandas/numpy safe).
OPEN_ENDED_NEXT_RELEASE = pd.Timestamp("2262-01-01 00:00:00")

# Timeframes that should NOT be forced onto US business days.
_SKIP_BUSINESS_DAY_ADJUST = {
    "continuous_market",
    "minute",
    "hourly",
}


# ==========================================================
# Business day utilities
# ==========================================================

def is_business_day(dt) -> bool:
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.weekday() >= 5:
        return False
    return dt.date() not in _US_HOLIDAY_DATES


def next_business_day(dt) -> datetime:
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    result = dt
    while not is_business_day(result):
        result += timedelta(days=1)
    return result


def adjust_to_business_day(dt: datetime) -> datetime:
    if not is_business_day(dt):
        dt = next_business_day(dt)
    return dt


def parse_release_time(release_time_str: str) -> tuple:
    try:
        hour, minute = map(int, release_time_str.split(":"))
        return hour, minute
    except (ValueError, AttributeError):
        return 16, 0


# ==========================================================
# Calculate release_datetime
# ==========================================================

def calculate_release_datetime(
    date: pd.Timestamp,
    meta: Dict[str, Any],
    first_release: Optional[pd.Timestamp] = None,
) -> datetime:
    """
    Calculate when an observation becomes available.

    Priority:
        1. first_release (ALFRED first publication date), if valid
        2. Heuristic from observation date + lag + release_time

    Heuristic note:
        FRED observation ``date`` is the period label (already the right
        calendar anchor). We do NOT add an extra week/month/quarter on top
        of that label — that previously double-counted delay. Only
        ``release_lag_days`` shifts availability after the observation date.
    """
    time_str = meta.get("release_time", "16:00")
    hour, minute = parse_release_time(time_str)
    timeframe = meta.get("release_timeframe", "daily")
    lag_days = int(meta.get("release_lag_days", 0) or 0)

    # ---- 1. Prefer real first-publication date (vintage) ----
    if first_release is not None and not pd.isna(first_release):
        fr = pd.Timestamp(first_release)
        # If only a date, attach configured release clock time
        if fr.hour == 0 and fr.minute == 0 and fr.second == 0:
            release_dt_local = datetime.combine(fr.date(), dt_time(hour, minute))
        else:
            release_dt_local = fr.to_pydatetime()
        if timeframe not in _SKIP_BUSINESS_DAY_ADJUST:
            release_dt_local = adjust_to_business_day(release_dt_local)
        return release_dt_local

    # ---- 2. Heuristic from observation date ----
    if isinstance(date, pd.Timestamp):
        obs = date.to_pydatetime()
    else:
        obs = date

    # Observation date is the period label. Availability = label + lag.
    # Special cases only when the dictionary explicitly asks for them.
    if timeframe == "next_business_day":
        base_date = next_business_day(obs)
    else:
        # daily / weekly / biweekly / monthly / quarterly / continuous / etc.
        # Period length is already encoded in how often FRED emits rows;
        # lag_days is the publication delay after the observation date.
        base_date = obs

    release_date = base_date + timedelta(days=lag_days)
    release_dt_local = datetime.combine(
        release_date.date() if hasattr(release_date, "date") else release_date,
        dt_time(hour, minute),
    )

    if timeframe not in _SKIP_BUSINESS_DAY_ADJUST and meta.get("timezone") != "UTC":
        # Continuous / UTC markets (e.g. gold) keep calendar days as-is.
        if meta.get("source") != "TwelveData":
            release_dt_local = adjust_to_business_day(release_dt_local)

    return release_dt_local


def _repair_interval_order(records: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Ensure release times are non-decreasing and next_release > release.
    Drops empty / inverted intervals after sorting by release.
    """
    if records.empty:
        return records

    records = records.sort_values("release_datetime").reset_index(drop=True)

    # Force non-decreasing releases (keep last value on ties by dropping earlier dups)
    records = records.drop_duplicates(subset=["release_datetime"], keep="last")
    records = records.sort_values("release_datetime").reset_index(drop=True)

    # Rebuild next_release from the sorted sequence (authoritative)
    releases = records["release_datetime"].tolist()
    nexts = []
    for i in range(len(releases)):
        if i + 1 < len(releases):
            nxt = releases[i + 1]
            # Guard: if equal or inverted (shouldn't after sort), nudge
            if pd.Timestamp(nxt) <= pd.Timestamp(releases[i]):
                nxt = pd.Timestamp(releases[i]) + pd.Timedelta(seconds=1)
            nexts.append(nxt)
        else:
            nexts.append(OPEN_ENDED_NEXT_RELEASE)
    records["next_release_datetime"] = nexts

    # Drop zero-length intervals
    valid = pd.to_datetime(records["next_release_datetime"]) > pd.to_datetime(
        records["release_datetime"]
    )
    n_bad = (~valid).sum()
    if n_bad:
        logger.warning(
            f"  {feature}: dropped {n_bad} empty/inverted interval(s)"
        )
    records = records.loc[valid].reset_index(drop=True)
    return records


# ==========================================================
# Main: Build availability records
# ==========================================================

def build_availability(
    raw_df: pd.DataFrame,
    features: Dict[str, Any],
) -> pd.DataFrame:
    """Build availability records from raw data."""

    logger.info("=" * 60)
    logger.info("Building availability records")
    logger.info("=" * 60)

    all_records = []
    total_features = len(features)

    for idx, (feature, meta) in enumerate(features.items(), 1):
        if feature not in raw_df.columns:
            continue

        first_rel_col = f"{feature}__first_release"
        cols = ["date", feature]
        has_vintage = first_rel_col in raw_df.columns
        if has_vintage:
            cols.append(first_rel_col)

        series = raw_df[cols].dropna(subset=[feature])
        if len(series) == 0:
            continue

        logger.info(
            f"  [{idx}/{total_features}] {feature}: {len(series):,} records"
            + (" [vintage first_release]" if has_vintage else " [heuristic lag]")
        )
        logger.info(f"    First date: {series['date'].iloc[0]}")
        logger.info(f"    Last date:  {series['date'].iloc[-1]}")

        series = series.sort_values("date").reset_index(drop=True)
        dates = series["date"]
        values = series[feature]
        first_rels = (
            pd.to_datetime(series[first_rel_col], errors="coerce")
            if has_vintage
            else pd.Series([pd.NaT] * len(series))
        )

        records = []
        for i in range(len(dates)):
            current_date = dates.iloc[i]
            current_value = values.iloc[i]
            fr = first_rels.iloc[i] if has_vintage else None

            release_dt = calculate_release_datetime(
                current_date, meta, first_release=fr
            )

            # Provisional next; rebuilt after sort in _repair_interval_order
            if i + 1 < len(dates):
                next_fr = first_rels.iloc[i + 1] if has_vintage else None
                next_release_dt = calculate_release_datetime(
                    dates.iloc[i + 1], meta, first_release=next_fr
                )
            else:
                next_release_dt = OPEN_ENDED_NEXT_RELEASE

            records.append(
                {
                    "feature": feature,
                    "release_datetime": release_dt,
                    "next_release_datetime": next_release_dt,
                    "value": current_value,
                }
            )

        feature_df = pd.DataFrame(records)
        feature_df = _repair_interval_order(feature_df, feature)
        if len(feature_df) == 0:
            logger.warning(f"  {feature}: no valid intervals after repair")
            continue
        all_records.append(feature_df)

    if not all_records:
        raise RuntimeError("No availability records created.")

    result = pd.concat(all_records, ignore_index=True)
    result = result.sort_values(["feature", "release_datetime"]).reset_index(
        drop=True
    )

    # Ensure datetime dtypes for clean CSV round-trip
    result["release_datetime"] = pd.to_datetime(result["release_datetime"])
    result["next_release_datetime"] = pd.to_datetime(
        result["next_release_datetime"]
    )

    logger.info(f"Total records: {len(result):,}")
    logger.info(f"Features: {result['feature'].nunique()}")

    return result


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    import sys
    import os

    # Config-free artifact: path may come from env, but catalog is core FEATURES
    # only (no include/exclude, no macro_features_extra). TRAIN selects features
    # later when building the minute grid (fsv_03).
    data_dir = os.environ.get("MACRO_DATA_DIR") or os.environ.get("FSV_DATA_DIR", "dataset_cache")

    RAW_FILE = f"{data_dir}/macro_raw.csv"
    OUTPUT_FILE = f"{data_dir}/macro_value_intervals.csv"

    print("\n" + "=" * 60)
    print("FSV Availability Builder (config-independent intervals CSV)")
    print("=" * 60)

    print(f"\nLoading raw FSV from {RAW_FILE}...")
    try:
        raw = pd.read_csv(RAW_FILE, parse_dates=["date"])
        # Parse any __first_release columns as dates
        for c in raw.columns:
            if c.endswith("__first_release"):
                raw[c] = pd.to_datetime(raw[c], errors="coerce")
        print(f"Loaded {len(raw):,} rows, {len(raw.columns)} columns")
        print(f"Date range: {raw['date'].min()} to {raw['date'].max()}")
    except FileNotFoundError:
        print(f"ERROR: File not found: {RAW_FILE}")
        sys.exit(1)

    print("\nBuilding availability records...")
    # Core dictionary only — do not pass training config.
    catalog = get_features(None, apply_filters=False)
    print(f"Using {len(catalog)} core feature metas for interval build")
    availability = build_availability(raw, catalog)

    print(f"\nSaving availability data to {OUTPUT_FILE}...")
    os.makedirs(data_dir, exist_ok=True)
    availability.to_csv(OUTPUT_FILE, index=False, float_format="%.8g", mode="w")

    print(f"\n✅ Saved: {OUTPUT_FILE}")
    print(f"   Records: {len(availability):,}")

    print("\nSample:")
    sample_cols = [
        "feature",
        "release_datetime",
        "next_release_datetime",
        "value",
    ]
    print(availability[sample_cols].head(15).to_string())

    # Sanity: no null next_release
    n_null_next = availability["next_release_datetime"].isna().sum()
    print(f"\nNull next_release_datetime: {n_null_next} (should be 0)")

    print("\n" + "=" * 60)
