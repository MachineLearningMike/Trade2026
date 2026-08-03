"""
fsv_02_find_value_intervals.py

Financial State Vector Availability Engine.

Input:
    fsv_raw.csv
    fsv_dictionary.py

Output:
    fsv_value_intervals.csv

Converts each raw row (date, feature, value) to:
    (feature, release_datetime, next_release_datetime, value)

The raw dataset is never modified.
"""

import time
import logging
from datetime import datetime, time as dt_time, timedelta
from typing import Optional, Dict, Any

import pandas as pd
import pytz
from pandas.tseries.holiday import USFederalHolidayCalendar

from fsv_dictionary import FEATURES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==========================================================
# Constants
# ==========================================================

_US_HOLIDAYS = USFederalHolidayCalendar().holidays(
    start="1990-01-01",
    end="2030-12-31"
)


# ==========================================================
# Business day utilities
# ==========================================================

def is_business_day(dt) -> bool:
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.weekday() >= 5:
        return False
    return dt.date() not in _US_HOLIDAYS


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
) -> datetime:
    """Calculate release datetime for a value on a given date."""
    
    if isinstance(date, pd.Timestamp):
        obs = date.to_pydatetime()
    else:
        obs = date
    
    timeframe = meta.get("release_timeframe", "daily")
    lag_days = meta.get("release_lag_days", 0)
    time_str = meta.get("release_time", "16:00")
    
    # Step 1: Apply timeframe
    if timeframe == "same_day" or timeframe == "daily":
        base_date = obs
    elif timeframe == "next_business_day":
        base_date = next_business_day(obs)
    elif timeframe == "weekly":
        base_date = obs + timedelta(days=7)
        base_date = next_business_day(base_date)
    elif timeframe == "monthly":
        base_date = pd.Timestamp(obs) + pd.offsets.MonthBegin(1)
        base_date = base_date.to_pydatetime()
    elif timeframe == "quarterly":
        quarter_start = pd.Timestamp(obs).to_period("Q").start_time
        quarter_end = quarter_start + pd.offsets.QuarterEnd(1)
        base_date = quarter_end.to_pydatetime()
    elif timeframe == "continuous_market":
        base_date = obs
    else:
        base_date = obs
    
    # Step 2: Add lag_days
    release_date = base_date + timedelta(days=lag_days)
    
    # Step 3: Add time
    hour, minute = parse_release_time(time_str)
    release_dt_local = datetime.combine(release_date.date(), dt_time(hour, minute))
    
    # Step 4: Adjust to business day
    release_dt_local = adjust_to_business_day(release_dt_local)
    
    return release_dt_local


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
    
    start_time = time.time()
    all_records = []
    total_features = len(features)
    
    for idx, (feature, meta) in enumerate(features.items(), 1):
        if feature not in raw_df.columns:
            continue
        
        # Get series with non-null values
        series = raw_df[["date", feature]].dropna()
        
        if len(series) == 0:
            continue
        
        logger.info(f"  [{idx}/{total_features}] {feature}: {len(series):,} records...")
        logger.info(f"    First date: {series['date'].iloc[0]}")
        logger.info(f"    Last date: {series['date'].iloc[-1]}")
        
        dates = series["date"].sort_values().reset_index(drop=True)
        values = series[feature].reset_index(drop=True)
        
        records = []
        
        for i in range(len(dates)):
            current_date = dates.iloc[i]
            current_value = values.iloc[i]
            
            # Calculate release_datetime for current value
            release_dt = calculate_release_datetime(current_date, meta)
            
            # Calculate next_release_datetime
            if i + 1 < len(dates):
                next_date = dates.iloc[i + 1]
                next_release_dt = calculate_release_datetime(next_date, meta)
            else:
                next_release_dt = None
            
            records.append({
                "feature": feature,
                "release_datetime": release_dt,
                "next_release_datetime": next_release_dt,
                "value": current_value,
            })
        
        all_records.append(pd.DataFrame(records))
    
    if not all_records:
        raise RuntimeError("No availability records created.")
    
    result = pd.concat(all_records, ignore_index=True)
    result = result.sort_values(["feature", "release_datetime"]).reset_index(drop=True)
    
    logger.info(f"Total records: {len(result):,}")
    logger.info(f"Features: {result['feature'].nunique()}")
    
    return result


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    
    import sys
    import os
    
    RAW_FILE = "fsv_raw.csv"
    OUTPUT_FILE = "fsv_value_intervals.csv"
    
    print("\n" + "=" * 60)
    print("FSV Availability Builder")
    print("=" * 60)
    
    # Load raw data
    print(f"\nLoading raw FSV from {RAW_FILE}...")
    try:
        raw = pd.read_csv(RAW_FILE, parse_dates=["date"])
        print(f"Loaded {len(raw):,} rows, {len(raw.columns)} columns")
        print(f"Date range: {raw['date'].min()} to {raw['date'].max()}")
    except FileNotFoundError:
        print(f"ERROR: File not found: {RAW_FILE}")
        sys.exit(1)
    
    # Build availability
    print("\nBuilding availability records...")
    availability = build_availability(raw, FEATURES)
    
    # Save results (overwrite existing file)
    print(f"\nSaving availability data to {OUTPUT_FILE}...")
    availability.to_csv(OUTPUT_FILE, index=False, float_format="%.8g", mode='w')
    
    print(f"\n✅ Saved: {OUTPUT_FILE}")
    print(f"   Records: {len(availability):,}")
    
    # Show sample
    print("\nSample:")
    sample_cols = ["feature", "release_datetime", "next_release_datetime", "value"]
    print(availability[sample_cols].head(15).to_string())
    
    print("\n" + "=" * 60)