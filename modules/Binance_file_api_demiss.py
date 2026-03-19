"""
Binance Historical Data Downloader - STRICT UTC ONLY with MONTHLY PRIORITY and API FALLBACK
- All datetimes are treated as UTC
- Naive datetimes are assumed to be UTC
- No local time conversion functions
- No argparse - just function calls
- PRIORITY: Monthly files > Daily files (uses monthly files when available)
- CLEANUP: After downloading a monthly file, delete all daily files for that month
- FALLBACK: Binance API for recent data when files aren't available
- TRACKING: Counts newly downloaded files
"""

import os
import sys
import time
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import glob

import numpy as np
import pandas as pd
from binance.client import Client

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Supported intervals
SUPPORTED_INTERVALS = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']

# Binance data repository base URL
BASE_URL_DOWNLOAD = "https://data.binance.vision/"

# Default configuration
Config = {
    'STORE_DIRECTORY': 'd:/binance_data',
    'api_key': '',  # Add your API key if needed
    'api_secret': '',  # Add your API secret if needed
}

# UTC timezone constant
UTC = timezone.utc

# Global download counter
_download_counter = 0
_deleted_files_counter = 0

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# Cache for interval to milliseconds conversion
_interval_cache = {}

def intervalToMilliseconds(interval):
    """
    Convert a Binance interval string to milliseconds with caching
    """
    if interval in _interval_cache:
        return _interval_cache[interval]
    
    ms = None
    seconds_per_unit = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60
    }

    unit = interval[-1]
    if unit in seconds_per_unit:
        try:
            ms = int(interval[:-1]) * seconds_per_unit[unit] * 1000
        except ValueError:
            pass
    
    _interval_cache[interval] = ms
    return ms


def ensure_utc(dt, param_name="datetime"):
    """
    Ensure datetime is UTC-aware.
    If naive, assume it's UTC and add timezone.
    If aware but not UTC, raise error.
    """
    if dt is None:
        return None
        
    if dt.tzinfo is None:
        # Naive datetime - assume UTC
        return dt.replace(tzinfo=UTC)
    elif dt.tzinfo != UTC:
        # Has timezone but not UTC - error
        raise ValueError(f"{param_name} must be UTC. Got {dt.tzinfo}")
    else:
        # Already UTC
        return dt


def get_current_day_start_utc(dt_utc):
    """Get start of day for given UTC datetime"""
    dt_utc = ensure_utc(dt_utc, "dt_utc")
    return datetime(dt_utc.year, dt_utc.month, dt_utc.day).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC
    )


def get_next_day_start_utc(dt_utc):
    """Get start of next day in UTC"""
    day_start = get_current_day_start_utc(dt_utc)
    return day_start + timedelta(days=1)


def get_current_month_start_utc(dt_utc):
    """Get start of month for given UTC datetime"""
    dt_utc = ensure_utc(dt_utc, "dt_utc")
    return datetime(dt_utc.year, dt_utc.month, 1).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC
    )


def get_next_month_start_utc(dt_utc):
    """Get start of next month in UTC"""
    start_in_month = get_current_month_start_utc(dt_utc)
    # Add 32 days to ensure next month
    next_month = start_in_month + timedelta(days=32)
    return get_current_month_start_utc(next_month)


def get_base_path(datetime_utc, dataType, symbol, interval, file_type='daily'):
    """
    Get base path and filename for Binance data repository files
    file_type: 'monthly' or 'daily'
    """
    datetime_utc = ensure_utc(datetime_utc, "datetime_utc")
    
    if dataType == 'klines':
        if file_type == 'monthly':
            base_path = "data/spot/monthly/klines/{}/{}/".format(symbol.upper(), interval)
            file_name = "{}-{}-{}.zip".format(symbol.upper(), interval, datetime_utc.strftime("%Y-%m"))
        else:  # daily
            base_path = "data/spot/daily/klines/{}/{}/".format(symbol.upper(), interval)
            file_name = "{}-{}-{}.zip".format(symbol.upper(), interval, datetime_utc.strftime("%Y-%m-%d"))
    else:
        raise Exception('Invalid dataType: {}'.format(dataType))

    return base_path, file_name


def get_download_url(file_url):
    """Get full download URL for Binance data repository"""
    return "{}{}".format(BASE_URL_DOWNLOAD, file_url)


def get_file_location(datetime_utc, dataType, symbol, interval, file_type='daily'):
    """Get local file paths for data storage"""
    base_folder, file_name = get_base_path(datetime_utc, dataType, symbol, interval, file_type=file_type)
    save_folder = os.path.join(os.environ.get('STORE_DIRECTORY', Config['STORE_DIRECTORY']), base_folder)
    save_path = os.path.join(save_folder, file_name)
    return base_folder, file_name, save_folder, save_path


def parse_datalines_to_datalines(dataType, datalines):
    """Convert string data to proper numeric types"""
    if dataType == 'klines' and datalines:
        result = []
        for row in datalines:
            row_list = list(row)
            # Convert timestamps from microseconds to milliseconds by dividing by 1000
            new_row = [
                int(float(row_list[0]) / 1000),     # Open time (convert from micro to milli)
                float(row_list[1]),                  # Open
                float(row_list[2]),                  # High
                float(row_list[3]),                  # Low
                float(row_list[4]),                  # Close
                float(row_list[5]),                  # Volume
                int(float(row_list[6]) / 1000),      # Close time (convert from micro to milli)
                float(row_list[7]),                  # Quote asset volume
                int(float(row_list[8])),             # Number of trades
                float(row_list[9]),                   # Taker buy base asset volume
                float(row_list[10]),                  # Taker buy quote asset volume
                float(row_list[11])                   # Ignore
            ]
            result.append(new_row)
        return result
    return datalines


def parse_api_datalines(dataType, datalines):
    """
    Parse data from API (timestamps in milliseconds)
    Keep timestamps as is (no division)
    """
    if dataType == 'klines' and datalines:
        result = []
        for row in datalines:
            row_list = list(row)
            # Timestamps are already in milliseconds
            new_row = [
                int(row_list[0]),                    # Open time (already ms)
                float(row_list[1]),                   # Open
                float(row_list[2]),                   # High
                float(row_list[3]),                   # Low
                float(row_list[4]),                   # Close
                float(row_list[5]),                   # Volume
                int(row_list[6]),                     # Close time (already ms)
                float(row_list[7]),                   # Quote asset volume
                int(row_list[8]),                      # Number of trades
                float(row_list[9]),                    # Taker buy base asset volume
                float(row_list[10]),                   # Taker buy quote asset volume
                float(row_list[11])                    # Ignore
            ]
            result.append(new_row)
        return result
    return datalines


def snap_sort_deredundant_in_opening_time(dynarray, interval_mili):
    """
    Clean and sort array, remove duplicates, ensure proper intervals
    """
    if dynarray is None or len(dynarray) == 0:
        return dynarray
        
    org = dynarray.shape[0]
    
    # Normalize timestamps (now in milliseconds)
    dynarray[:, 0] = (dynarray[:, 0] / interval_mili).astype(int) * interval_mili
    if interval_mili <= 60000:
        dynarray[:, 6] = (dynarray[:, 6] / interval_mili).astype(int) * interval_mili

    if dynarray.shape[0] >= 2:
        # Sort by opening time
        sort_idx = np.argsort(dynarray[:, 0])
        dynarray = dynarray[sort_idx]
        
        # Find and remove duplicates/out-of-order
        delta = dynarray[1:, 0] - dynarray[:-1, 0]
        valid_mask = np.insert(delta >= interval_mili, 0, True)
        dynarray = dynarray[valid_mask]

    delta = org - dynarray.shape[0]
    if delta > 0:
        print("R-{} ".format(delta), end="")

    return dynarray


def delete_daily_files_for_month(symbol, interval, year, month, data_dir):
    """
    Delete all daily files for a specific month after monthly file is downloaded
    """
    global _deleted_files_counter
    
    daily_base_path = os.path.join(data_dir, "data/spot/daily/klines", symbol.upper(), interval)
    
    if not os.path.exists(daily_base_path):
        return
    
    # Pattern for all days in this month
    pattern = os.path.join(daily_base_path, f"{symbol.upper()}-{interval}-{year:04d}-{month:02d}-*.csv")
    daily_files = glob.glob(pattern)
    
    # Also delete zip files
    zip_pattern = os.path.join(daily_base_path, f"{symbol.upper()}-{interval}-{year:04d}-{month:02d}-*.zip")
    zip_files = glob.glob(zip_pattern)
    
    all_files = daily_files + zip_files
    
    for file_path in all_files:
        try:
            os.remove(file_path)
            print(f"    🗑️ Deleted daily file: {os.path.basename(file_path)}")
            _deleted_files_counter += 1
        except Exception as e:
            print(f"    ⚠️ Failed to delete {file_path}: {e}")
    
    if all_files:
        print(f"    ✅ Deleted {len(all_files)} daily files for {year:04d}-{month:02d}")


def download_file(current_utc, dataType, symbol, interval, file_type='daily'):
    """
    Download file from Binance data repository
    Returns path to CSV file if successful, None otherwise
    """
    global _download_counter
    
    current_utc = ensure_utc(current_utc, "current_utc")
    
    base_folder, file_name, save_folder, save_path = get_file_location(
        current_utc, dataType, symbol, interval, file_type=file_type
    )
    download_path = os.path.join(base_folder, file_name)

    csv_path = str(Path(save_path).with_suffix('.csv'))
    
    # If CSV already exists, return it
    if os.path.exists(csv_path):
        print(f"  File already exists: {csv_path}")
        return csv_path

    # Create directory if needed
    if not os.path.exists(save_folder):
        Path(save_folder).mkdir(parents=True, exist_ok=True)

    # Download the zip file if it doesn't exist
    if not os.path.exists(save_path):
        try:
            download_url = get_download_url(download_path)
            print(f"  Downloading {file_type} file: {download_url}")
            dl_file = urllib.request.urlopen(download_url)
            length = dl_file.getheader('content-length')
            if length:
                length = int(length)
                blocksize = max(4096, length // 100)

            with open(save_path, 'wb') as out_file:
                dl_progress = 0
                while True:
                    buf = dl_file.read(blocksize)
                    if not buf:
                        break
                    dl_progress += len(buf)
                    out_file.write(buf)
                    if length:
                        done = round(50 * dl_progress / length)
                        sys.stdout.write(f"\r  Download progress: {done*2}%")
                        sys.stdout.flush()
            print("\n  Download complete.")
            
            # Increment download counter
            _download_counter += 1
            print(f"  📥 New download #{_download_counter}: {file_name}")

        except urllib.error.HTTPError as e:
            print(f"  {file_type} file not found: {download_url} - {e}")
            return None
        except Exception as e:
            print(f"  Download error: {e}")
            return None

    # Extract the zip file
    if os.path.exists(save_path) and not os.path.exists(csv_path):
        try:
            print(f"  Extracting: {save_path}")
            import zipfile
            with zipfile.ZipFile(save_path, 'r') as zip_ref:
                zip_ref.extractall(save_folder)
            print("  Extraction complete.")
            
            if os.path.exists(csv_path):
                # If this is a monthly file, delete all daily files for this month
                if file_type == 'monthly':
                    year = current_utc.year
                    month = current_utc.month
                    delete_daily_files_for_month(symbol, interval, year, month, Config['STORE_DIRECTORY'])
                
                return csv_path
            else:
                print(f"  CSV not found after extraction: {csv_path}")
                return None
                
        except zipfile.BadZipFile:
            print(f"  Bad zip file: {save_path}")
            try:
                os.remove(save_path)
            except:
                pass
            return None
        except Exception as e:
            print(f"  Extraction error: {e}")
            return None

    return csv_path if os.path.exists(csv_path) else None


def read_csv_file(file_path, start_mili=None, end_mili=None):
    """
    Read CSV file and optionally filter by time range.
    Auto-detects timestamp format:
    - If timestamps are > 1e13 (16+ digits), they're microseconds → divide by 1000
    - If timestamps are around 1e12-1e13 (13 digits), they're milliseconds → keep as is
    """
    try:
        print(f"  Reading file: {os.path.basename(file_path)}")
        df = pd.read_csv(file_path, header=None)
        
        if df.empty:
            print("  File is empty")
            return None
            
        data = df.to_numpy()
        print(f"  File contains {len(data)} total rows")
        
        # Check timestamp format using first row
        raw_timestamp = float(data[0, 0])
        
        # Auto-detect format
        if raw_timestamp > 1e13:  # More than 10 trillion = microseconds (16 digits)
            print(f"  Detected microsecond timestamps → dividing by 1000")
            data[:, 0] = data[:, 0] / 1000  # Open time
            data[:, 6] = data[:, 6] / 1000  # Close time
        else:  # Already milliseconds (13 digits)
            print(f"  Detected millisecond timestamps → keeping as is")
            # Just ensure they're integers
            data[:, 0] = data[:, 0].astype(np.int64)
            data[:, 6] = data[:, 6].astype(np.int64)
        
        print(f"  First timestamp (ms): {data[0,0]}")
        print(f"  Last timestamp (ms): {data[-1,0]}")
        
        # Filter by time range if provided
        if start_mili is not None and end_mili is not None:
            print(f"  Filtering from {start_mili} to {end_mili}")
            mask = (data[:, 0] >= start_mili - 1) & (data[:, 0] <= end_mili + 1)
            filtered = data[mask]
            print(f"  Got {len(filtered)} rows in time range")
            return filtered
        else:
            return data
        
    except Exception as e:
        print(f"  Error reading file {file_path}: {e}")
        return None


def fetch_from_api(symbol, interval, start_utc, end_utc):
    """
    Fetch kline data from Binance API when files are not available
    API returns timestamps in milliseconds (keep as is)
    """
    print(f"\n  📡 Falling back to Binance API for {symbol} {interval}")
    print(f"  API range: {start_utc} to {end_utc}")
    
    try:
        # Create Binance client
        client = Client(Config['api_key'], Config['api_secret'])
        
        # Convert to milliseconds
        start_mili = int(start_utc.timestamp() * 1000)
        end_mili = int(end_utc.timestamp() * 1000)
        
        print(f"  Start timestamp (ms): {start_mili}")
        print(f"  End timestamp (ms): {end_mili}")
        
        # Calculate expected rows
        interval_mili = intervalToMilliseconds(interval)
        expected_rows = (end_mili - start_mili) // interval_mili + 1
        print(f"  Expected rows: {expected_rows}")
        
        all_klines = []
        current_start = start_mili
        
        # Binance API limit is 1000 candles per request
        while current_start < end_mili:
            print(f"    Fetching chunk starting at {datetime.fromtimestamp(current_start/1000, UTC)}")
            
            klines = client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=current_start,
                endTime=end_mili,
                limit=1000
            )
            
            if not klines:
                print(f"    No data returned for this chunk")
                break
                
            all_klines.extend(klines)
            print(f"    Got {len(klines)} rows in this chunk")
            print(f"    First timestamp in chunk: {klines[0][0]} ({datetime.fromtimestamp(klines[0][0]/1000, UTC)})")
            print(f"    Last timestamp in chunk: {klines[-1][0]} ({datetime.fromtimestamp(klines[-1][0]/1000, UTC)})")
            
            # Move to next chunk
            last_candle_time = klines[-1][0]
            current_start = last_candle_time + interval_mili
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        if all_klines:
            print(f"    ✓ Got {len(all_klines)} total rows from API")
            
            # Parse API data (timestamps already in milliseconds)
            rows = parse_api_datalines('klines', all_klines)
            data = np.array(rows)
            
            print(f"    After parsing: {len(data)} rows")
            print(f"    Parsed timestamps: {data[0,0]} to {data[-1,0]}")
            
            # Filter to exact time range
            mask = (data[:, 0] >= start_mili) & (data[:, 0] <= end_mili)
            filtered_data = data[mask]
            
            print(f"    After filtering: {len(filtered_data)} rows in range")
            
            if len(filtered_data) != expected_rows:
                print(f"    ⚠ Warning: Got {len(filtered_data)} rows, expected {expected_rows}")
            
            return filtered_data
        else:
            print(f"    ✗ No data from API")
            return None
            
    except Exception as e:
        print(f"    ✗ API error: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_files_for_month_no_api(symbol, interval, start_in_month, end_in_month):
    """
    Get data for a specific month from files only (monthly first, then daily)
    No API fallback here - this is pure file collection
    """
    month_data = []
    month_start_utc = get_current_month_start_utc(start_in_month)
    now_utc = datetime.now(UTC)
    
    print(f"\nProcessing month from files: {month_start_utc.strftime('%Y-%m')}")
    
    # TRY MONTHLY FILE FIRST
    monthly_file_path = None
    monthly_file_found = False
    
    base_folder, file_name, save_folder, save_path = get_file_location(
        month_start_utc, 'klines', symbol, interval, file_type='monthly'
    )
    monthly_csv_path = str(Path(save_path).with_suffix('.csv'))
    
    if os.path.exists(monthly_csv_path):
        print(f"  Found monthly file: {monthly_csv_path}")
        monthly_file_found = True
        monthly_file_path = monthly_csv_path
    else:
        # Try to download monthly file (only if it's old enough to exist)
        days_ago = (now_utc - month_start_utc).days
        if days_ago >= 3:
            print(f"  Monthly file not found locally, attempting download...")
            monthly_csv_path = download_file(month_start_utc, 'klines', symbol, interval, file_type='monthly')
            if monthly_csv_path and os.path.exists(monthly_csv_path):
                monthly_file_found = True
                monthly_file_path = monthly_csv_path
    
    # Use monthly file if found
    if monthly_file_found and monthly_file_path:
        print(f"  ✅ Using monthly file for {month_start_utc.strftime('%Y-%m')}")
        
        file_start = max(month_start_utc, start_in_month)
        file_end = min(get_next_month_start_utc(month_start_utc) - timedelta(milliseconds=1), end_in_month)
        
        start_mili = int(file_start.timestamp() * 1000)
        end_mili = int(file_end.timestamp() * 1000)
        
        data = read_csv_file(monthly_file_path, start_mili, end_mili)
        
        if data is not None and len(data) > 0:
            print(f"    ✓ Got {len(data)} rows from monthly file")
            month_data.append(data)
            return month_data, True
    
    # FALL BACK TO DAILY FILES
    print(f"  ⚠ Monthly file not available, trying daily files...")
    
    current_day = get_current_day_start_utc(max(start_in_month, month_start_utc))
    month_end_day = get_current_day_start_utc(min(end_in_month, get_next_month_start_utc(month_start_utc) - timedelta(milliseconds=1)))
    
    days_processed = 0
    
    while current_day <= month_end_day:
        day_end = min(current_day + timedelta(days=1) - timedelta(milliseconds=1), end_in_month)
        
        if current_day <= day_end:
            print(f"\n  Checking daily file for {current_day.strftime('%Y-%m-%d')}...")
            
            base_folder, file_name, save_folder, save_path = get_file_location(
                current_day, 'klines', symbol, interval, file_type='daily'
            )
            csv_path = str(Path(save_path).with_suffix('.csv'))
            
            if not os.path.exists(csv_path):
                day_days_ago = (now_utc - current_day).days
                if day_days_ago >= 2:
                    print(f"    Daily file not found locally, downloading...")
                    csv_path = download_file(current_day, 'klines', symbol, interval, file_type='daily')
            
            if csv_path and os.path.exists(csv_path):
                start_mili = int(max(current_day, start_in_month).timestamp() * 1000)
                end_mili = int(day_end.timestamp() * 1000)
                
                data = read_csv_file(csv_path, start_mili, end_mili)
                
                if data is not None and len(data) > 0:
                    print(f"    ✓ Got {len(data)} rows from daily file")
                    month_data.append(data)
                    days_processed += 1
        
        current_day += timedelta(days=1)
    
    if days_processed > 0:
        print(f"\n  ✅ Used {days_processed} daily files for {month_start_utc.strftime('%Y-%m')}")
        return month_data, True
    
    print(f"\n  ✗ No file data found for {month_start_utc.strftime('%Y-%m')}")
    return month_data, False

def get_historical_klines(symbol, interval, start_utc, end_utc=None, use_api_fallback=True):
    """
    Main function to get historical kline data from files, with API fallback for recent data
    
    Parameters:
    - symbol: str (e.g., 'BTCUSDT')
    - interval: str (e.g., '1h', '1d')
    - start_utc: datetime (UTC - can be naive or aware)
    - end_utc: datetime (UTC - can be naive or aware, optional, defaults to now UTC)
    - data_dir: str (directory to store downloaded files)
    - use_api_fallback: bool (whether to use API when files are not available)
    
    Returns:
    - numpy array of kline data with columns: [open, high, low, close, volume, quote_volume, trades, taker_buy_base, taker_buy_quote]
    - or None if failed
    """
    
    global _download_counter, _deleted_files_counter
    _download_counter = 0  # Reset counter at start of each request
    _deleted_files_counter = 0
    
    # Ensure inputs are UTC
    start_utc = ensure_utc(start_utc, "start_utc")
    if end_utc is None:
        end_utc = datetime.now(UTC)
    else:
        end_utc = ensure_utc(end_utc, "end_utc")
    
    # Don't fetch future data
    now_utc = datetime.now(UTC)
    if end_utc > now_utc:
        print(f"  Warning: end_utc is in the future, using now UTC")
        end_utc = now_utc
    
    # Calculate expected number of rows BEFORE normalization
    interval_mili = intervalToMilliseconds(interval)
    total_duration_ms = (end_utc.timestamp() - start_utc.timestamp()) * 1000
    expected_rows_before_norm = int(total_duration_ms / interval_mili) + 1
    
    # Normalize to interval boundaries
    start_utc_norm = datetime.fromtimestamp(
        (start_utc.timestamp() * 1000 // interval_mili) * interval_mili / 1000,
        tz=UTC
    )
    end_utc_norm = datetime.fromtimestamp(
        (end_utc.timestamp() * 1000 // interval_mili) * interval_mili / 1000,
        tz=UTC
    )
    
    # Calculate expected rows AFTER normalization
    total_duration_ms_norm = (end_utc_norm.timestamp() - start_utc_norm.timestamp()) * 1000
    expected_rows = int(total_duration_ms_norm / interval_mili)
    
    print(f"\nFetching {symbol} {interval} from {start_utc_norm} to {end_utc_norm}")
    print(f"  Expected rows: {expected_rows} (before normalization: {expected_rows_before_norm})")
    
    all_data = []
    months_processed_file = 0
    months_processed_api = 0
    
    # FIRST: Collect all available file data (monthly + daily)
    current_month_start = get_current_month_start_utc(start_utc_norm)
    end_month_start = get_current_month_start_utc(end_utc_norm)
    
    while current_month_start <= end_month_start:
        # current_month_end = min(
        #     get_next_month_start_utc(current_month_start) - timedelta(milliseconds=1),
        #     end_utc_norm
        # )
        current_month_end = get_next_month_start_utc(current_month_start) - timedelta(milliseconds=1)
        
        # Get data for this month from files only (no API here)
        month_data, from_files = get_files_for_month_no_api(
            symbol, interval, 
            max(current_month_start, start_utc_norm), 
            min(current_month_end, end_utc_norm), 
        )
        
        if month_data:
            all_data.extend(month_data)
            months_processed_file += 1
        
        current_month_start = get_next_month_start_utc(current_month_start)
    
    # If we have no data at all, nothing to do
    if not all_data:
        print("\n❌ No data found in files")
        if not use_api_fallback:
            return None, False
        else:
            # No files, will try API for entire range
            combined = np.array([])
            file_rows = 0
    else:
        # Combine file data to see what we have
        combined = np.vstack(all_data)
        combined = combined[np.argsort(combined[:, 0])]
        
        # Remove duplicate timestamps
        _, unique_idx = np.unique(combined[:, 0], return_index=True)
        combined = combined[unique_idx]
        
        last_file_timestamp = combined[-1, 0]
        last_file_time = datetime.fromtimestamp(last_file_timestamp / 1000, tz=UTC)
        first_file_timestamp = combined[0, 0]
        first_file_time = datetime.fromtimestamp(first_file_timestamp / 1000, tz=UTC)
        
        print(f"\n📁 Data from files: {first_file_time} to {last_file_time}")
        print(f"  Rows from files: {len(combined)}")
        
        # ====================================================================
        # VERIFY AND FILL MISSING DATA IN FILE DATA
        # ====================================================================
        if combined.size > 0:
            combined, nFilled, filled_ok = verify_and_fill_data(
                symbol, interval, start_utc_norm, end_utc_norm, combined
            )
            if nFilled > 0:
                print(f"  ✓ Filled {nFilled} missing rows in file data")
        
        # Check if files already cover the entire requested range
        if first_file_timestamp <= int(start_utc_norm.timestamp() * 1000) and last_file_timestamp >= int(end_utc_norm.timestamp() * 1000):
            print(f"✅ Files cover entire requested range")
            
            # Verify row count
            if len(combined) == expected_rows:
                print(f"  ✓ Row count matches expected: {expected_rows}")
            else:
                print(f"  ⚠ Row count mismatch: got {len(combined)}, expected {expected_rows}")
            
            print(f"\n{'='*60}")
            print(f"DOWNLOAD SUMMARY:")
            print(f"{'='*60}")
            print(f"  📥 New files downloaded: {_download_counter}")
            print(f"  🗑️ Daily files deleted: {_deleted_files_counter}")
            
            # Filter to exact requested range
            start_mili = int(start_utc_norm.timestamp() * 1000)
            end_mili = int(end_utc_norm.timestamp() * 1000)
            mask = (combined[:, 0] >= start_mili) & (combined[:, 0] <= end_mili)
            combined = combined[mask]
            
            result = np.delete(combined, [0, 6], axis=1)
            return np.float32(result), True
    
    # SECOND: If API fallback is enabled and we need more recent data
    if use_api_fallback:
        # Determine where files left off
        if all_data:
            # combined already exists from above
            last_file_timestamp = combined[-1, 0]
            api_start = datetime.fromtimestamp(last_file_timestamp / 1000, tz=UTC) + timedelta(milliseconds=1)
            file_rows = len(combined)
        else:
            # No files at all, start from beginning
            api_start = start_utc_norm
            combined = np.array([])
            file_rows = 0
        
        # Only fetch API data if there's a gap to fill
        if api_start < end_utc_norm:
            print(f"\n📡 Fetching remaining data from API: {api_start} to {end_utc_norm}")
            
            api_data = fetch_from_api(symbol, interval, api_start, end_utc_norm)
            
            if api_data is not None and len(api_data) > 0:
                print(f"  ✓ Got {len(api_data)} rows from API")
                months_processed_api += 1
                
                # Combine with file data
                if combined.size > 0:
                    combined = np.vstack([combined, api_data])
                else:
                    combined = api_data
                
                # Sort and clean combined data
                combined = combined[np.argsort(combined[:, 0])]
                _, unique_idx = np.unique(combined[:, 0], return_index=True)
                combined = combined[unique_idx]
    
    # Final processing
    if combined.size == 0:
        print("\n❌ No data found")
        return None, False
    
    # Sort and clean (one more time to be safe)
    combined = combined[np.argsort(combined[:, 0])]
    _, unique_idx = np.unique(combined[:, 0], return_index=True)
    combined = combined[unique_idx]
    
    # ====================================================================
    # FINAL VERIFICATION AND FILLING (in case API data had gaps)
    # ====================================================================
    if combined.size > 0:
        combined, nFilled, filled_ok = verify_and_fill_data(
            symbol, interval, start_utc_norm, end_utc_norm, combined
        )
        
        if nFilled > 0:
            print(f"  ✓ Filled {nFilled} missing rows in combined data")
    
    # Verify total rows
    total_rows = len(combined)
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"{'='*60}")
    print(f"Total rows collected: {total_rows}")
    print(f"Expected rows: {expected_rows}")
    
    if total_rows == expected_rows:
        print(f"  ✓ Row count matches expected")
    else:
        print(f"  ⚠ Row count mismatch: got {total_rows}, expected {expected_rows}")
        if total_rows < expected_rows:
            missing = expected_rows - total_rows
            print(f"     Missing {missing} rows")
        else:
            extra = total_rows - expected_rows
            print(f"     Extra {extra} rows")
    
    if months_processed_file > 0:
        print(f"  ✓ From files: {months_processed_file} month(s) ({file_rows if 'file_rows' in locals() else 'unknown'} rows)")
    if months_processed_api > 0:
        print(f"  ✓ From API: {months_processed_api} chunk(s) ({len(api_data) if 'api_data' in locals() else 0} rows)")
    print(f"  📥 New files downloaded: {_download_counter}")
    print(f"  🗑️ Daily files deleted: {_deleted_files_counter}")
    
    # Filter to exact requested range one last time
    start_mili = int(start_utc_norm.timestamp() * 1000)
    end_mili = int(end_utc_norm.timestamp() * 1000)
    mask = (combined[:, 0] >= start_mili) & (combined[:, 0] <= end_mili)
    combined = combined[mask]
    
    # Remove timestamp columns
    data = np.delete(combined, [0, 6], axis=1)
    data = np.float32(data)
    
    return data, total_rows == expected_rows

def get_historical_klines_batch(symbols, interval, start_utc, end_utc=None, data_dir='./binance_data', use_api_fallback=True):
    """Get historical kline data for multiple symbols"""
    results = {}
    total_downloads = 0
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Processing {symbol}")
        print(f"{'='*60}")
        
        try:
            data, _ = get_historical_klines(symbol, interval, start_utc, end_utc, data_dir, use_api_fallback)
            results[symbol] = data
            if data is not None:
                print(f"\n✓ {symbol}: {data.shape[0]} rows")
            else:
                print(f"\n✗ {symbol}: No data")
        except Exception as e:
            print(f"\n✗ {symbol}: Error - {e}")
            import traceback
            traceback.print_exc()
            results[symbol] = None
    
    return results


def is_one_incremental(arr):
    """Check if array values are consecutive integers"""
    if len(arr) <= 1:
        return True
    arr = np.asarray(arr)
    return np.all(arr[1:] - arr[:-1] == 1)


def find_missing_intervals(dynarray, interval_mili, start_mili, end_mili):
    """
    Find gaps in the data where timestamps are not consecutive
    
    Returns list of (gap_start, gap_end, expected_rows) tuples
    """
    if dynarray is None or len(dynarray) == 0:
        return [(start_mili, end_mili, (end_mili - start_mili) // interval_mili + 1)]
    
    missing_intervals = []
    
    # Sort by timestamp just in case
    dynarray = dynarray[np.argsort(dynarray[:, 0])]
    
    # Check gap at the beginning
    first_ts = dynarray[0, 0]
    if first_ts > start_mili:
        expected = (first_ts - start_mili) // interval_mili
        if expected > 0:
            missing_intervals.append((start_mili, first_ts - interval_mili, expected))
            print(f"    📍 Gap at beginning: missing {expected} rows from {start_mili} to {first_ts - interval_mili}")
    
    # Check gaps in the middle
    for i in range(len(dynarray) - 1):
        current_ts = dynarray[i, 0]
        next_ts = dynarray[i + 1, 0]
        expected_next = current_ts + interval_mili
        
        if next_ts > expected_next:
            gap_start = current_ts + interval_mili
            gap_end = next_ts - interval_mili
            expected_rows = (next_ts - current_ts) // interval_mili - 1
            missing_intervals.append((gap_start, gap_end, expected_rows))
            print(f"    📍 Gap in middle: missing {expected_rows} rows from {gap_start} to {gap_end}")
    
    # Check gap at the end
    last_ts = dynarray[-1, 0]
    if last_ts < end_mili:
        expected = (end_mili - last_ts) // interval_mili
        if expected > 0:
            missing_intervals.append((last_ts + interval_mili, end_mili, expected))
            print(f"    📍 Gap at end: missing {expected} rows from {last_ts + interval_mili} to {end_mili}")
    
    return missing_intervals


def fill_missing_intervals(symbol, interval, missing_intervals, interval_mili, surrounding_data=None):
    """
    Fill missing intervals by fetching from API first, then interpolating if needed
    """
    nCreated = 0
    all_filled_data = []
    
    for i, (gap_start, gap_end, expected_rows) in enumerate(missing_intervals):
        expected_rows = int(round(expected_rows))  # Convert to integer
        print(f"\n    🔍 Filling gap {i+1}/{len(missing_intervals)}: {gap_start} to {gap_end} ({expected_rows} rows)")
        
        # Convert timestamps to datetimes for API call
        gap_start_utc = datetime.fromtimestamp(gap_start / 1000, tz=UTC)
        gap_end_utc = datetime.fromtimestamp(gap_end / 1000, tz=UTC)
        
        # Try to fetch from API first
        api_data = fetch_from_api(symbol, interval, gap_start_utc, gap_end_utc)
        
        if api_data is not None and len(api_data) > 0:
            print(f"      ✓ Fetched {len(api_data)} rows from API")
            all_filled_data.append(api_data)
            nCreated += len(api_data)
        else:
            # API failed, create interpolated data
            print(f"      ⚠ API fetch failed, creating interpolated data for {expected_rows} rows")
            
            # Create interpolated rows
            interpolated = np.zeros((expected_rows, 12), dtype=float)
            
            # Set timestamps
            for j in range(expected_rows):
                interpolated[j, 0] = gap_start + j * interval_mili  # Open time
                interpolated[j, 6] = interpolated[j, 0] + interval_mili - 1  # Close time
            
            # Set default values for OHLCV (all zeros - will be filled later if needed)
            interpolated[:, 1:5] = 0  # OHLC
            interpolated[:, 5] = 0     # Volume
            interpolated[:, 7:11] = 0  # Quote fields
            interpolated[:, 11] = -1    # Mark as interpolated
            
            all_filled_data.append(interpolated)
            nCreated += expected_rows
            print(f"      ✓ Created {expected_rows} interpolated rows")
    
    return all_filled_data, nCreated


def verify_and_fill_data(symbol, interval, start_utc, end_utc, combined):
    """
    Verify data completeness and fill any missing points
    """
    if combined.size == 0:
        print("  ⚠ No data to verify")
        return combined, 0, False
    
    interval_mili = intervalToMilliseconds(interval)
    start_mili = int(start_utc.timestamp() * 1000)
    end_mili = int(end_utc.timestamp() * 1000)
    
    print(f"\n🔍 Verifying data completeness for {symbol} {interval}")
    print(f"  Range: {start_mili} to {end_mili}")
    print(f"  Current rows: {len(combined)}")
    
    # Sort by timestamp
    combined = combined[np.argsort(combined[:, 0])]
    
    # Find missing intervals
    missing = find_missing_intervals(combined, interval_mili, start_mili, end_mili)
    
    if not missing:
        print(f"  ✓ No gaps found")
        return combined, 0, True
    
    print(f"\n📊 Found {len(missing)} gaps to fill:")
    for gap_start, gap_end, expected in missing:
        expected = int(round(expected))
        start_dt = datetime.fromtimestamp(gap_start/1000, tz=UTC)
        end_dt = datetime.fromtimestamp(gap_end/1000, tz=UTC)
        print(f"    Gap: {start_dt} to {end_dt} ({expected} rows)")
    
    # Fill missing intervals
    filled_data_list, nCreated = fill_missing_intervals(
        symbol, interval, missing, interval_mili, combined
    )
    
    if filled_data_list:
        # Combine with original data
        all_data = [combined] + filled_data_list
        combined = np.vstack(all_data)
        
        # Sort by timestamp
        combined = combined[np.argsort(combined[:, 0])]
        
        # Remove duplicates
        _, unique_idx = np.unique(combined[:, 0], return_index=True)
        combined = combined[unique_idx]
        
        print(f"\n  ✓ After filling: total rows = {len(combined)}")
    
    # Final verification
    expected_total = (end_mili - start_mili) // interval_mili # + 1     # end is treated as exclusive
    if len(combined) == expected_total:
        print(f"  ✓ Data is now complete ({len(combined)} rows)")
        return combined, nCreated, True
    else:
        print(f"  ✗ Data still incomplete: have {len(combined)}, expected {expected_total}")
        return combined, nCreated, False
    

# ============================================================================
# EXAMPLE USAGE (no argparse)
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("BINANCE HISTORICAL DATA DOWNLOADER - MONTHLY PRIORITY WITH API FALLBACK")
    print("="*70)
    
    # Example 1: Historical data from files
    print("\n" + "-"*70)
    print("EXAMPLE 1: Historical data (2024-01-01 to 2024-03-31) - should use files")
    print("-"*70)
    
    start_utc = datetime(2018, 1, 1, 0, 0, 0)
    end_utc = datetime(2026, 3, 15, 0, 0, 0)
    
    data, successful = get_historical_klines(
        symbol='BTCUSDT',
        interval='1m',
        start_utc=start_utc,
        end_utc=end_utc,
        use_api_fallback=True
    )
    
    if data is not None:
        print(f"\n✅ successful {successful}. Got {data.shape[0]} rows")
        print(f"First row (OHLCV): {data[0]}")
    else:
        print(f"\n❌ Failed")
