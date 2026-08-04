"""
fsv_03_query_value_age.py

Query FSV value intervals for a given datetime.

Returns: (value, decay)
    - value: The data value
    - decay: 1 / 2 ** (age / timeframe)

One state structure for all features.
State is reset for each feature, then maintained across minutes within that feature.
Results are organized by feature: [ [ (value, decay) for minutes ] for features ]
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union, Tuple, List, Generator
import logging
import pytz
from functools import lru_cache
import hashlib
import json
import os

from fsv_dictionary import FEATURES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==========================================================
# TIMEFRAME CONSTANTS (in minutes)
# ==========================================================

TIMEFRAME_MINUTES = {
    "minute": 1,
    "hourly": 60,
    "daily": 60 * 24,
    "weekly": 60 * 24 * 7,
    "biweekly": 60 * 24 * 14,
    "monthly": 60 * 24 * 30,
    "quarterly": 60 * 24 * 30 * 3,
}


def get_timeframe_minutes(feature: str) -> int:
    """Get timeframe in minutes for a feature from fsv_dictionary.py."""
    meta = FEATURES.get(feature, {})
    timeframe_str = meta.get("release_timeframe", "daily")
    return TIMEFRAME_MINUTES.get(timeframe_str, 60 * 24)


def compute_decay(age_minutes: float, timeframe_minutes: int) -> float:
    """
    Compute decay weight: 1 / 2 ** (age / timeframe)
    """
    if age_minutes is None or timeframe_minutes is None or timeframe_minutes == 0:
        return 1.0
    return 1.0 / (2.0 ** (age_minutes / timeframe_minutes))


def generate_cache_filename(
    features: List[str],
    query_datetime_from: Union[str, datetime, pd.Timestamp],
    query_datetime_to: Union[str, datetime, pd.Timestamp],
    is_utc: bool,
    prefix: str = "fsv_value_decay",
) -> str:
    """
    Generate a short unique filename by hashing all parameters collectively.
    """
    if isinstance(query_datetime_from, pd.Timestamp):
        dt_from_str = query_datetime_from.strftime("%Y%m%d_%H%M%S")
    else:
        dt_from_str = str(query_datetime_from)
    
    if isinstance(query_datetime_to, pd.Timestamp):
        dt_to_str = query_datetime_to.strftime("%Y%m%d_%H%M%S")
    else:
        dt_to_str = str(query_datetime_to)
    
    features_str = ",".join(sorted(features))
    param_str = f"{features_str}|{dt_from_str}|{dt_to_str}|{is_utc}"
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    
    return f"{prefix}_{param_hash}.npy"


class FeatureState:
    """
    ONE state structure for ALL features.
    
    Reset for each feature, then maintained across minutes within that feature.
    """
    
    def __init__(self):
        self.idx = 0
        self.current_value = None
        self.initialized = False
        self.release = None
        self.next_release = None
        self.values = None
        self.timeframe = 0
    
    def set_intervals(self, intervals: Dict[str, np.ndarray]):
        """Set the intervals for the current feature."""
        self.release = intervals['release']
        self.next_release = intervals['next_release']
        self.values = intervals['value']
        self.timeframe = intervals['timeframe']
    
    def reset(self):
        """Reset state for a new feature."""
        self.idx = 0
        self.current_value = None
        self.initialized = False
    
    def advance_to(self, query_np: np.datetime64, query_dt_local: pd.Timestamp) -> Tuple[Optional[float], Optional[float]]:
        """
        Advance the sliding window to contain query_np.
        Returns (value, decay) if found, (None, None) otherwise.
        """
        release = self.release
        next_release = self.next_release
        values = self.values
        timeframe = self.timeframe
        
        if release is None:
            return None, None
        
        # If not initialized, find the first interval
        if not self.initialized:
            lo, hi = 0, len(release)
            while lo < hi:
                mid = (lo + hi) // 2
                if query_np < release[mid]:
                    hi = mid
                elif mid + 1 < len(release) and query_np >= next_release[mid]:
                    lo = mid + 1
                else:
                    self.idx = mid
                    self.current_value = values[mid]
                    self.initialized = True
                    break
            
            if not self.initialized:
                return None, None
        
        # If query is before current interval, restart search
        if query_np < release[self.idx]:
            lo, hi = 0, len(release)
            self.initialized = False
            while lo < hi:
                mid = (lo + hi) // 2
                if query_np < release[mid]:
                    hi = mid
                elif mid + 1 < len(release) and query_np >= next_release[mid]:
                    lo = mid + 1
                else:
                    self.idx = mid
                    self.current_value = values[mid]
                    self.initialized = True
                    break
            
            if not self.initialized:
                return None, None
        
        # Advance while we've passed the next_release_datetime
        while self.idx + 1 < len(release) and query_np >= next_release[self.idx]:
            self.idx += 1
            self.current_value = values[self.idx]
        
        # Check if we have a valid interval
        if self.idx < len(release) and query_np < next_release[self.idx]:
            # Compute decay
            release_val = release[self.idx]
            age_seconds = (query_dt_local - pd.Timestamp(release_val)).total_seconds()
            age_minutes = age_seconds / 60.0
            decay = compute_decay(age_minutes, timeframe)
            return self.current_value, decay
        
        self.initialized = False
        return None, None


class FSVValueLookup:
    """
    Query FSV value intervals for a given datetime.
    
    Returns: (value, decay)
    Results organized by feature: [ [ (value, decay) for minutes ] for features ]
    """
    
    def __init__(
        self,
        csv_file: str = "dataset_cache/fsv_value_intervals.csv",
        features_to_exclude: Optional[List[str]] = None,
        cache_size: int = 10000,
    ):
        """
        Initialize the FSVValueLookup.
        
        Args:
            csv_file: Path to the CSV file containing value intervals
            features_to_exclude: List of features to exclude from all queries
            cache_size: Size of the LRU cache for single queries
        """
        logger.info(f"Loading FSV value intervals from {csv_file}...")
        logger.info(f"  Cache size: {cache_size}")
        
        # Load the data
        self.df = pd.read_csv(csv_file)
        
        # Parse datetime columns
        self.df["release_datetime"] = pd.to_datetime(self.df["release_datetime"])
        self.df["next_release_datetime"] = pd.to_datetime(self.df["next_release_datetime"])
        
        # Sort by feature and release_datetime
        self.df = self.df.sort_values(["feature", "release_datetime"]).reset_index(drop=True)
        
        # Set features_to_exclude
        if features_to_exclude is None:
            features_to_exclude = []
        self.features_to_exclude = features_to_exclude
        
        # Build timezone map (returns dict) - includes all features
        self._timezone_map = self._build_timezone_map()
        
        # Build interval index (returns dict) - EXCLUDES features_to_exclude
        self._index = self._build_interval_index(self.features_to_exclude)
        self.features = list(self._index.keys())
        
        
        # ONE state structure for ALL features
        self._state = FeatureState()
        
        # LRU cache for single queries
        self._get_value_cached = lru_cache(maxsize=cache_size)(self._get_value_impl)
        
        logger.info(f"Loaded {len(self.df):,} records for {len(self._index)} features")
        logger.info(f"Features to exclude: {len(self.features_to_exclude)}")
        if self.features_to_exclude:
            logger.info(f"  Excluded: {self.features_to_exclude}")
    
    def _build_timezone_map(self) -> Dict[str, str]:
        """Build timezone map from fsv_dictionary.py for all features."""
        timezone_map: Dict[str, str] = {}
        for feature, meta in FEATURES.items():
            tz = meta.get("timezone", "America/New_York")
            timezone_map[feature] = tz
        return timezone_map
    
    def _build_interval_index(self, features_to_exclude: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Build interval index, EXCLUDING features in features_to_exclude.
        
        Args:
            features_to_exclude: List of feature names to exclude
        
        Structure:
            {
                feature: {
                    'release': np.array(datetime64),
                    'next_release': np.array(datetime64),
                    'value': np.array(float64),
                    'timeframe': int,
                }
            }
        
        Returns:
            Dict mapping feature name to its intervals (excluded features omitted)
        """
        exclude_set = set(features_to_exclude)
        index: Dict[str, Dict[str, np.ndarray]] = {}
        
        for feature in self.df["feature"].unique():
            if feature in exclude_set:
                continue
            
            mask = self.df["feature"] == feature
            feature_df = self.df.loc[mask, ["release_datetime", "next_release_datetime", "value"]]
            
            index[feature] = {
                'release': feature_df["release_datetime"].values.astype('datetime64[ns]'),
                'next_release': feature_df["next_release_datetime"].values.astype('datetime64[ns]'),
                'value': feature_df["value"].values.astype('float64'),
                'timeframe': get_timeframe_minutes(feature),
            }
        
        return index
    
    def _convert_to_local_timezone(
        self,
        query_dt: pd.Timestamp,
        feature: str,
    ) -> pd.Timestamp:
        """Convert UTC datetime to feature's local timezone."""
        tz_str = self._timezone_map.get(feature, "America/New_York")
        
        if query_dt.tz is None:
            query_dt = query_dt.tz_localize('UTC')
        elif query_dt.tz != pytz.UTC:
            query_dt = query_dt.tz_convert('UTC')
        
        tz = pytz.timezone(tz_str)
        return query_dt.tz_convert(tz).replace(tzinfo=None)
    
    def _prepare_query_dt_local(
        self,
        query_dt: pd.Timestamp,
        feature: str,
        is_utc: bool = False,
    ) -> pd.Timestamp:
        """Prepare query datetime in feature's local timezone."""
        if is_utc:
            return self._convert_to_local_timezone(query_dt, feature)
        return query_dt
    
    def _get_value_impl(
        self,
        feature: str,
        query_dt_str: str,
        is_utc: bool = False,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Actual implementation of get_value_decay (cached).
        """
        query_dt = pd.Timestamp(query_dt_str)
        query_dt_local = self._prepare_query_dt_local(query_dt, feature, is_utc)
        
        if feature not in self._index:
            return None, None
        
        intervals = self._index[feature]
        release_dt = intervals['release']
        next_release_dt = intervals['next_release']
        values = intervals['value']
        timeframe = intervals['timeframe']
        
        query_np = np.datetime64(query_dt_local)
        
        lo, hi = 0, len(release_dt)
        
        while lo < hi:
            mid = (lo + hi) // 2
            release_val = release_dt[mid]
            
            if query_np < release_val:
                hi = mid
            elif mid + 1 < len(release_dt) and query_np >= next_release_dt[mid]:
                lo = mid + 1
            else:
                age_seconds = (query_dt_local - pd.Timestamp(release_val)).total_seconds()
                age_minutes = age_seconds / 60.0
                decay = compute_decay(age_minutes, timeframe)
                return values[mid], decay
        
        return None, None
    
    def get_value_decay(
        self,
        feature: str,
        query_datetime: Union[str, datetime, pd.Timestamp],
        is_utc: bool = False,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Get (value, decay) for a feature at a specific datetime."""
        dt_str = str(query_datetime)
        return self._get_value_cached(feature, dt_str, is_utc)

    def get_arranged_features(self, features: Optional[List[str]] = None) -> List[str]:
        """
        Return the list of features to query.
        Only includes features that are in self._index (excluded ones are already filtered out).
        """
        if features is None:
            return [f for f in self.features if f not in self.features_to_exclude]
        
        return [f for f in self.features if f in features and f not in self.features_to_exclude]
    
    def get_value_decay_batch_accumulated(
        self,
        query_datetime_from: Union[str, datetime, pd.Timestamp],
        query_datetime_to: Union[str, datetime, pd.Timestamp],
        features: Optional[List[str]] = None,
        is_utc: bool = False,
    ) -> Tuple[List[List[Tuple[Optional[float], Optional[float]]]], Dict[str, int]]:
        """
        Returns results organized by feature: [ [ (value, decay) for minutes ] for features ]
        and miss counts per feature.
        
        ONE state structure for ALL features.
        State is reset for each feature, then maintained across minutes within that feature.
        """
        features = self.get_arranged_features(features)
        
        dt_from = pd.Timestamp(query_datetime_from)
        dt_to = pd.Timestamp(query_datetime_to)
        total_minutes = int((dt_to - dt_from).total_seconds() / 60)
        
        if total_minutes <= 0:
            return [], {feature: 0 for feature in features}
        
        logger.info(f"Getting values for {len(features)} features over {total_minutes} minutes")
        
        # Generate all minute datetimes ONCE
        minute_dt_strs = [(dt_from + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S") 
                         for i in range(total_minutes)]
        
        results_by_feature = []
        miss_counts = {feature: 0 for feature in features}
        
        # ONE state structure for ALL features
        state = self._state
        
        for fid, feature in enumerate(features):
            if feature not in self._index:
                results_by_feature.append([(None, None)] * total_minutes)
                miss_counts[feature] += total_minutes
                continue

            print(f"{fid+1}. Processing feature: {feature} ({len(minute_dt_strs)} minutes) ...")
            
            # ==========================================================
            # RESET STATE FOR THIS FEATURE (fresh start)
            # ==========================================================
            state.reset()
            state.set_intervals(self._index[feature])
            
            feature_results = []
            
            # Process ALL minutes for THIS feature
            for i, dt_str in enumerate(minute_dt_strs):
                query_dt = pd.Timestamp(dt_str)
                
                if is_utc:
                    tz_str = self._timezone_map.get(feature, "America/New_York")
                    tz = pytz.timezone(tz_str)
                    query_dt_local = query_dt.tz_localize('UTC').tz_convert(tz).replace(tzinfo=None)
                else:
                    query_dt_local = query_dt
                
                query_np = np.datetime64(query_dt_local)
                
                # state is maintained across minutes within this feature
                value, decay = state.advance_to(query_np, query_dt_local)
                if value is None:
                    miss_counts[feature] += 1
                feature_results.append((value, decay))
            
            results_by_feature.append(feature_results)
        
        return results_by_feature, miss_counts
    
    def get_features(self) -> list:
        return self.features
    
    def get_excluded_features(self) -> list:
        return self.features_to_exclude
    
    def get_date_range(self) -> tuple:
        return (self.df["release_datetime"].min(), self.df["release_datetime"].max())
    
    def clear_cache(self):
        self._get_value_cached.cache_clear()
        logger.info("Cache cleared")


def format_value(value: Optional[float]) -> str:
    if value is None:
        return "None"
    return f"{value:,.4f}"


def format_decay(decay: Optional[float]) -> str:
    if decay is None:
        return "None"
    return f"{decay:.4f}"

def get_value_age_array(config):

    import time
    import os
    
    print("=" * 60)
    print("FSV Value Lookup - With Cache")
    print("=" * 60)
    
    # Create cache directory
    data_dir = config.get("data_dir", "dataset_cache")
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialize lookup with excluded features
    features_to_exclude = config.get("features_to_exclude", [])
    
    lookup = FSVValueLookup(
        f"{data_dir}/fsv_value_intervals.csv",
        features_to_exclude=features_to_exclude,
        cache_size=10000,
    )
    
    print(f"\nExcluded features: {lookup.get_excluded_features()}")
    print(f"Available features: {len(lookup.get_features())}")
    
    # Define query parameters
    dt_from = config.get('binance_start_date', "2024-01-01 00:00:00")
    dt_to = config.get('binance_end_date', "2026-07-01 00:00:00")
    features = None
    features = lookup.get_arranged_features(features)
    is_utc = True
    
    print(f"\nFeatures to query: {features}, len={len(features)}")
    print(f"Time range: {dt_from} to {dt_to}")
    print(f"Is UTC: {is_utc}")
    
    # Generate cache filename
    cache_filename = generate_cache_filename(
        features=features,
        query_datetime_from=dt_from,
        query_datetime_to=dt_to,
        is_utc=is_utc,
    )
    cache_path = os.path.join(data_dir, cache_filename)
    
    print(f"\nCache file: {cache_path}")
    
    # Initialize variables
    arr = np.array([], dtype=np.float32)
    total_minutes = 0
    miss_counts = {}
    
    # ==========================================================
    # Check if cache exists, load or compute
    # ==========================================================
    
    if os.path.exists(cache_path):
        print("\nCache hit! Loading from file...")
        start = time.time()
        arr = np.load(cache_path)
        elapsed = time.time() - start
        print(f"Loaded array shape: {arr.shape}")
        print(f"Load time: {elapsed:.4f}s")
        
        # Load metadata if exists
        metadata_path = cache_path.replace('.npy', '_meta.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            miss_counts = metadata.get('miss_counts', {})
            total_minutes = metadata.get('total_minutes', arr.shape[1] if len(arr.shape) > 1 else 0)
        else:
            miss_counts = {}
            total_minutes = arr.shape[1] if len(arr.shape) > 1 else 0
    else:
        print("\nCache miss! Computing...")
        
        # Compute results
        start = time.time()
        results, miss_counts = lookup.get_value_decay_batch_accumulated(
            query_datetime_from=dt_from,
            query_datetime_to=dt_to,
            features=features,
            is_utc=is_utc,
        )
        compute_time = time.time() - start
        
        total_features = len(results)
        
        if total_features == 0:
            print("No features to process. Check your feature list.")
            arr = np.array([], dtype=np.float32)
        else:
            total_minutes = len(results[0]) if results else 0
            print(f"Results shape: {total_features} features x {total_minutes} minutes")
            print(f"Compute time: {compute_time:.4f}s")
            
            # Convert to numpy array
            print("\nConverting to numpy array...")
            start = time.time()
            arr = np.array(results, dtype=np.float32)
            convert_time = time.time() - start
            
            print(f"Array shape: {arr.shape}")
            print(f"Array dtype: {arr.dtype}")
            print(f"Convert time: {convert_time:.4f}s")
            
            # Save to cache
            print(f"\nSaving to cache: {cache_path}")
            np.save(cache_path, arr)
            
            # Save metadata with miss counts
            metadata = {
                'features': features,
                'total_minutes': total_minutes,
                'miss_counts': miss_counts,
                'dt_from': str(dt_from),
                'dt_to': str(dt_to),
                'is_utc': is_utc,
                'shape': arr.shape,
                'features_to_exclude': lookup.get_excluded_features(),
            }
            metadata_path = cache_path.replace('.npy', '_meta.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved metadata to: {metadata_path}")

    # ==========================================================
    # Null/NaN Value Analysis (only if array is not empty)
    # ==========================================================

    print("\n" + "=" * 60)
    print("Null/NaN Value Analysis")
    print("=" * 60)
    
    if arr.size == 0:
        print("\nArray is empty. No analysis to perform.")
        print("Please use a valid time range (dt_to > dt_from).")
    else:
        # Check for NaN values in the array
        nan_mask = np.isnan(arr)
        nan_count = nan_mask.sum()
        total_elements = arr.size
        
        print(f"\nTotal elements in array: {total_elements:,}")
        print(f"Total NaN values: {nan_count:,} ({nan_count / total_elements * 100:.2f}%)")
        
        # ==========================================================
        # Feature-level null analysis
        # ==========================================================
        
        print("\n" + "-" * 60)
        print("Feature-level Null Analysis")
        print("-" * 60)
        
        feature_null_stats = []
        
        for i, feature in enumerate(features):
            feature_nan_count = nan_mask[i, :, :].sum()
            feature_nan_pct = feature_nan_count / (arr.shape[1] * 2) * 100  # 2 = value + decay
            
            miss_count = miss_counts.get(feature, 0)
            miss_pct = miss_count / total_minutes * 100 if total_minutes > 0 else 0
            
            feature_null_stats.append({
                'feature': feature,
                'nan_count': feature_nan_count,
                'nan_pct': feature_nan_pct,
                'miss_count': miss_count,
                'miss_pct': miss_pct,
                'total_minutes': total_minutes,
            })
        
        # Sort by NaN percentage (highest first)
        feature_null_stats.sort(key=lambda x: x['nan_pct'], reverse=True)
        
        print(f"\n{'Feature':<30} {'NaN Count':<12} {'NaN %':<10} {'Miss Count':<12} {'Miss %':<10}")
        print("-" * 75)
        
        for stats in feature_null_stats:
            if stats['nan_count'] > 0 or stats['miss_count'] > 0:
                print(f"{stats['feature']:<30} {stats['nan_count']:<12} {stats['nan_pct']:<10.2f} {stats['miss_count']:<12} {stats['miss_pct']:<10.2f}")
        
        # ==========================================================
        # Features with 100% null values (completely missing)
        # ==========================================================
        
        fully_null_features = [s for s in feature_null_stats if s['nan_pct'] >= 100]
        if fully_null_features:
            print("\n" + "-" * 60)
            print("Features with 100% null values (completely missing):")
            for s in fully_null_features:
                print(f"  - {s['feature']}")
        
        # ==========================================================
        # Features with partial null values
        # ==========================================================
        
        partially_null = [s for s in feature_null_stats if 0 < s['nan_pct'] < 100]
        if partially_null:
            print("\n" + "-" * 60)
            print("Features with partial null values:")
            for s in partially_null[:10]:  # Show top 10
                print(f"  - {s['feature']}: {s['nan_pct']:.2f}% null")
            if len(partially_null) > 10:
                print(f"  ... and {len(partially_null) - 10} more features with partial nulls")
        
        # ==========================================================
        # Features with no null values
        # ==========================================================
        
        no_null_features = [s for s in feature_null_stats if s['nan_pct'] == 0]
        if no_null_features:
            print("\n" + "-" * 60)
            print(f"Features with no null values: {len(no_null_features)} out of {len(features)}")
            print(f"  {', '.join([s['feature'] for s in no_null_features[:10]])}")
            if len(no_null_features) > 10:
                print(f"  ... and {len(no_null_features) - 10} more")
        
        # ==========================================================
        # Show sample data
        # ==========================================================
        
        print("\n" + "=" * 60)
        print("Sample data (first 3 features, first 5 minutes)")
        print("=" * 60)
        
        for i in range(min(3, arr.shape[0])):
            feature = features[i]
            print(f"\n  {feature}:")
            for j in range(min(5, arr.shape[1])):
                dt = pd.Timestamp(dt_from) + timedelta(minutes=j)
                value = arr[i, j, 0]
                decay = arr[i, j, 1]
                print(f"    Minute {j} ({dt.strftime('%H:%M:%S')}): value={value:.4f}, decay={decay:.4f}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    shape = arr.shape   # (num_features, num_minutes, 2)
    reshaped = arr.transpose(0, 2, 1).reshape(-1, shape[1])     # Reshape to (num_features * 2, num_minutes)

    return reshaped, lookup.features



if __name__ == "__main__":

    from config import get_config

    config = get_config()

    arr, macro_feature_str_list = get_value_age_array(config)
