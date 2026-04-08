import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


def Download_Candles(symbol: str, interval: str, start_utc: str, end_utc: str, save_folder: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Download candle data from Binance Futures
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT', 'ETH/USDT')
        interval: Timeframe (e.g., '1m', '5m', '15m', '1h', '4h', '1d', '5s', '1s')
        start_utc: Start time in UTC (format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS')
        end_utc: End time in UTC (format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS')
        save_folder: Directory to save the CSV file
    
    Returns:
        Tuple of (DataFrame, filepath) or (None, error_message)
    """
    
    # Valid intervals for Binance Futures
    valid_intervals = {
        '1s': '1s', '5s': '5s', '15s': '15s', '30s': '30s',
        '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
        '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'
    }
    
    if interval not in valid_intervals:
        return None, f"Invalid interval. Choose from: {list(valid_intervals.keys())}"
    
    # Parse dates
    try:
        if ' ' in start_utc:
            start_dt = datetime.strptime(start_utc, '%Y-%m-%d %H:%M:%S')
        else:
            start_dt = datetime.strptime(start_utc, '%Y-%m-%d')
        
        if ' ' in end_utc:
            end_dt = datetime.strptime(end_utc, '%Y-%m-%d %H:%M:%S')
        else:
            end_dt = datetime.strptime(end_utc, '%Y-%m-%d')
    except ValueError as e:
        return None, f"Date parsing error: {e}. Use format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'"
    
    # Ensure end_dt is at end of day if only date provided
    if ' ' not in end_utc:
        end_dt = end_dt.replace(hour=23, minute=59, second=59)
    
    if start_dt >= end_dt:
        return None, "Start time must be before end time"
    
    # Create save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    
    # Initialize exchange
    # Use futures for intervals <= 1 minute, spot for others
    if interval in ['1s', '5s', '15s', '30s']:
        exchange = ccxt.binance({
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
        })
    else:
        exchange = ccxt.binance({
            'enableRateLimit': True,
        })
    
    # Convert to timestamp
    since = exchange.parse8601(start_dt.strftime('%Y-%m-%dT%H:%M:%SZ'))
    end_timestamp = exchange.parse8601(end_dt.strftime('%Y-%m-%dT%H:%M:%SZ'))
    
    all_ohlcv = []
    max_retries = 3
    
    print(f"\n{'='*70}")
    print(f"📥 DOWNLOADING CANDLE DATA")
    print(f"{'='*70}")
    print(f"Symbol:      {symbol}")
    print(f"Interval:    {interval}")
    print(f"Start:       {start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"End:         {end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Save folder: {save_folder}")
    print(f"{'='*70}\n")
    
    total_candles_expected = (end_dt - start_dt).total_seconds() / _interval_to_seconds(interval)
    print(f"Expected candles: ~{int(total_candles_expected):,}")
    
    while since < end_timestamp:
        retries = 0
        success = False
        
        while retries < max_retries and not success:
            try:
                # Fetch candles
                ohlcv = exchange.fetch_ohlcv(
                    symbol, 
                    timeframe=interval, 
                    since=since, 
                    limit=1000
                )
                
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1  # Move to next candle
                
                # Progress indicator
                candles_so_far = len(all_ohlcv)
                progress = min(100, (candles_so_far / total_candles_expected) * 100)
                print(f"  Downloaded {candles_so_far:6,} candles ({progress:5.1f}%)...", end='\r')
                
                success = True
                time.sleep(0.1)  # Rate limit protection
                
            except ccxt.RateLimitExceeded as e:
                retries += 1
                wait_time = retries * 2
                print(f"\n  Rate limit hit. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                
            except ccxt.BadSymbol as e:
                return None, f"Invalid symbol: {symbol}. Check Binance for available symbols."
                
            except Exception as e:
                retries += 1
                print(f"\n  Error (attempt {retries}/{max_retries}): {str(e)[:100]}")
                time.sleep(2)
        
        if not success:
            break
    
    print()  # New line after progress
    
    if not all_ohlcv:
        return None, "No data downloaded. Check your parameters."
    
    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume'
    ])
    
    # Convert timestamp to datetime
    df['open_time'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close_time'] = df['open_time'] + pd.Timedelta(seconds=_interval_to_seconds(interval))
    
    # Calculate additional columns
    df['quote_volume'] = df['close'] * df['volume']
    df['trades_count'] = 0
    df['taker_buy_base_volume'] = 0
    df['taker_buy_quote_volume'] = 0
    df['ignore'] = 0
    
    # Sort by time
    df = df.sort_values('open_time').reset_index(drop=True)
    
    # Generate filename
    safe_symbol = symbol.replace('/', '_')
    start_str = start_dt.strftime('%Y%m%d_%H%M%S')
    end_str = end_dt.strftime('%Y%m%d_%H%M%S')
    filename = f"{safe_symbol}_{interval}_{start_str}_to_{end_str}.csv"
    filepath = os.path.join(save_folder, filename)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    
    # Calculate statistics
    file_size = os.path.getsize(filepath) / 1024  # KB
    actual_start = df['open_time'].min()
    actual_end = df['open_time'].max()
    missing_candles = int(total_candles_expected) - len(df)
    
    print(f"\n{'='*70}")
    print(f"✅ DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"Total candles:  {len(df):,}")
    print(f"Missing:        {max(0, missing_candles):,}")
    print(f"Date range:     {actual_start} to {actual_end}")
    print(f"File size:      {file_size:.2f} KB")
    print(f"Saved to:       {filepath}")
    print(f"{'='*70}\n")
    
    return df, filepath


def _interval_to_seconds(interval: str) -> int:
    """Convert interval string to seconds"""
    unit = interval[-1]
    value = int(interval[:-1]) if len(interval) > 1 else 1
    
    multipliers = {
        's': 1,
        'm': 60,
        'h': 3600,
        'd': 86400,
        'w': 604800,
        'M': 2592000  # Approximate
    }
    
    return value * multipliers.get(unit, 60)


def fill_missing_candles(df: pd.DataFrame, expected_interval_seconds: int = None) -> pd.DataFrame:
    """
    Fill missing candles with dummy candles marked with ignore = -2
    
    Args:
        df: DataFrame with candle data (must have 'open_time' column)
        expected_interval_seconds: Expected time between candles (auto-detected if None)
    
    Returns:
        DataFrame with all candles filled
    """
    if df.empty:
        return df
    
    # Auto-detect interval if not provided
    if expected_interval_seconds is None and len(df) > 1:
        time_diffs = df['open_time'].diff().dt.total_seconds()
        expected_interval_seconds = int(time_diffs.median())
    
    if expected_interval_seconds is None:
        expected_interval_seconds = 60  # Default to 1 minute
    
    # Ensure open_time is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['open_time']):
        df['open_time'] = pd.to_datetime(df['open_time'])
    
    # Sort by time
    df = df.sort_values('open_time').reset_index(drop=True)
    
    # Create full time range
    start_time = df['open_time'].min()
    end_time = df['open_time'].max()
    
    full_time_range = pd.date_range(
        start=start_time,
        end=end_time,
        freq=f'{expected_interval_seconds}S'
    )
    
    # Create full DataFrame
    full_df = pd.DataFrame({'open_time': full_time_range})
    merged_df = full_df.merge(df, on='open_time', how='left')
    
    # Identify missing candles
    missing_mask = merged_df['open'].isna()
    missing_count = missing_mask.sum()
    
    if missing_count > 0:
        # Fill missing candles with dummy values
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 
                    'close_time', 'trades_count', 'taker_buy_base_volume', 
                    'taker_buy_quote_volume']:
            if col in merged_df.columns:
                merged_df.loc[missing_mask, col] = 0 if col != 'close_time' else pd.NaT
        
        # Mark missing candles
        if 'ignore' not in merged_df.columns:
            merged_df['ignore'] = 0
        merged_df.loc[missing_mask, 'ignore'] = -2
        
        # Set close_time for missing candles
        if 'close_time' in merged_df.columns:
            merged_df.loc[missing_mask, 'close_time'] = merged_df.loc[missing_mask, 'open_time'] + pd.Timedelta(seconds=expected_interval_seconds)
        
        print(f"  Filled {missing_count} missing candles ({missing_count/len(merged_df)*100:.1f}%)")
    
    return merged_df


# ========== USAGE EXAMPLES ==========

if __name__ == "__main__":
    
    # Example 1: Download 1-minute data for the last 7 days
    print("\n" + "="*70)
    print("EXAMPLE 1: Download 1-minute BTC/USDT data")
    print("="*70)
    
    df, path = Download_Candles(
        symbol='BTC/USDT',
        interval='1m',
        start_utc='2024-12-01',
        end_utc='2024-12-08',
        save_folder='./data/binance/'
    )
    
    if df is not None:
        print(f"\n📊 Data preview:")
        print(df[['open_time', 'open', 'high', 'low', 'close', 'volume']].head(10))
        
        # Fill missing candles
        df_filled = fill_missing_candles(df)
        
        # Save filled version
        filled_path = path.replace('.csv', '_filled.csv')
        df_filled.to_csv(filled_path, index=False)
        print(f"\n✅ Filled data saved to: {filled_path}")
    
    
    # Example 2: Download 5-second data (Futures only)
    print("\n" + "="*70)
    print("EXAMPLE 2: Download 5-second BTC/USDT data (Futures)")
    print("="*70)
    
    df_5sec, path_5sec = Download_Candles(
        symbol='BTC/USDT',
        interval='5s',
        start_utc='2024-12-08 00:00:00',
        end_utc='2024-12-08 01:00:00',  # Just 1 hour for testing
        save_folder='./data/binance/'
    )
    
    if df_5sec is not None:
        print(f"\n📊 5-Second data preview:")
        print(df_5sec[['open_time', 'open', 'close']].head(20))
    
    
    # Example 3: Download multiple symbols
    print("\n" + "="*70)
    print("EXAMPLE 3: Download multiple symbols")
    print("="*70)
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    for sym in symbols:
        df, path = Download_Candles(
            symbol=sym,
            interval='1h',
            start_utc='2024-12-01',
            end_utc='2024-12-08',
            save_folder='./data/binance/'
        )
        if df is not None:
            print(f"  ✓ {sym}: {len(df)} candles")
    
    
    # Example 4: Download with date range validation
    print("\n" + "="*70)
    print("EXAMPLE 4: Error handling example")
    print("="*70)
    
    df, error = Download_Candles(
        symbol='INVALID/PAIR',
        interval='1m',
        start_utc='2024-12-01',
        end_utc='2024-12-08',
        save_folder='./data/'
    )
    
    if df is None:
        print(f"❌ Error: {error}")
    
    print("\n✅ All examples completed!")