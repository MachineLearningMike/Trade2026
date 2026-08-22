"""
fsv_03_query_value_age.py

Query FSV value intervals for a given datetime.

Returns: (value, decay)
    - value: The data value
    - decay: 1 / 2 ** (age / timeframe)

Interval model (half-open):
    release_datetime <= t < next_release_datetime

The last interval uses a far-future next_release (from fsv_02) so the
latest observation remains available after its release.

One state structure for all features.
State is reset for each feature, then maintained across minutes within that feature.
Results are organized by feature: [ [ (value, decay) for minutes ] for features ]
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional, Dict, Any, Union, Tuple, List

import numpy as np
import pandas as pd
import pytz

from fsv_dictionary import FEATURES, get_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
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

# Must match fsv_02 open-ended sentinel in spirit (any large future is fine)
OPEN_ENDED_NEXT_RELEASE = np.datetime64("2262-01-01T00:00:00")


def get_timeframe_minutes(
    feature: str,
    catalog: Optional[Dict[str, Any]] = None,
) -> int:
    """Get timeframe in minutes from the active (or core) feature catalog."""
    cat = catalog if catalog is not None else FEATURES
    meta = cat.get(feature, {})
    timeframe_str = meta.get("release_timeframe", "daily")
    return TIMEFRAME_MINUTES.get(timeframe_str, 60 * 24)


def compute_decay(age_minutes: float, timeframe_minutes: int) -> float:
    """
    Compute decay weight: 1 / 2 ** (age / timeframe)
    """
    if age_minutes is None or timeframe_minutes is None or timeframe_minutes == 0:
        return 1.0
    if age_minutes < 0:
        # Should not happen if interval membership is enforced
        age_minutes = 0.0
    return 1.0 / (2.0 ** (age_minutes / timeframe_minutes))


def file_fingerprint(path: str) -> str:
    """
    Content-based fingerprint of the intervals CSV.

    Uses MD5 of file bytes (not mtime). Re-running fsv_02 with the same
    output keeps the same cache key; only real content changes bust it.

    Falls back to size-only if the file cannot be read.
    """
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()[:12]
    except OSError:
        try:
            st = os.stat(path)
            return hashlib.md5(f"size:{st.st_size}".encode()).hexdigest()[:12]
        except OSError:
            return "missing"


def generate_cache_filename(
    features: List[str],
    query_datetime_from: Union[str, datetime, pd.Timestamp],
    query_datetime_to: Union[str, datetime, pd.Timestamp],
    is_utc: bool,
    features_to_exclude: Optional[List[str]] = None,
    intervals_fingerprint: str = "",
    prefix: str = "fsv_value_decay",
) -> str:
    """
    Generate a short unique filename by hashing all parameters that affect
    the cached array contents.
    """
    if isinstance(query_datetime_from, pd.Timestamp):
        dt_from_str = query_datetime_from.strftime("%Y%m%d_%H%M%S")
    else:
        dt_from_str = str(query_datetime_from)

    if isinstance(query_datetime_to, pd.Timestamp):
        dt_to_str = query_datetime_to.strftime("%Y%m%d_%H%M%S")
    else:
        dt_to_str = str(query_datetime_to)

    features_str = ",".join(features)  # preserve order (order matters for array layout)
    excl = ",".join(sorted(features_to_exclude or []))
    param_str = (
        f"{features_str}|excl={excl}|{dt_from_str}|{dt_to_str}|"
        f"{is_utc}|iv={intervals_fingerprint}"
    )
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

    return f"{prefix}_{param_hash}.npy"


def find_interval_index(
    release: np.ndarray,
    next_release: np.ndarray,
    query_np: np.datetime64,
) -> int:
    """
    Find index i such that release[i] <= query_np < next_release[i].

    Returns -1 if no interval contains query_np.

    Uses binary search on release times, then validates membership with
    the half-open upper bound. This is the single source of truth for
    both single-query and batch (FeatureState) paths.
    """
    n = len(release)
    if n == 0:
        return -1

    # First index with release > query → candidate is one before
    # searchsorted side='right' gives insertion point after equal releases
    i = int(np.searchsorted(release, query_np, side="right") - 1)
    if i < 0:
        return -1

    # Walk forward if duplicate/adjacent edge cases leave us before a valid window
    while i + 1 < n and query_np >= next_release[i]:
        i += 1

    if i < 0 or i >= n:
        return -1

    # Half-open: [release, next_release)
    if query_np < release[i]:
        return -1
    if not (query_np < next_release[i]):
        return -1

    return i


class FeatureState:
    """
    Sliding pointer over one feature's intervals for sequential queries.

    Reset for each feature, then advanced monotonically across minutes.
    Uses the same membership rule as find_interval_index.
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
        self.release = intervals["release"]
        self.next_release = intervals["next_release"]
        self.values = intervals["value"]
        self.timeframe = intervals["timeframe"]

    def reset(self):
        """Reset state for a new feature."""
        self.idx = 0
        self.current_value = None
        self.initialized = False

    def advance_to(
        self, query_np: np.datetime64, query_dt_local: pd.Timestamp
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Advance the sliding window to contain query_np.
        Returns (value, decay) if found, (None, None) otherwise.
        """
        release = self.release
        next_release = self.next_release
        values = self.values
        timeframe = self.timeframe

        if release is None or len(release) == 0:
            return None, None

        # Cold start or query jumped backward: binary locate
        if (not self.initialized) or query_np < release[self.idx]:
            i = find_interval_index(release, next_release, query_np)
            if i < 0:
                self.initialized = False
                return None, None
            self.idx = i
            self.current_value = values[i]
            self.initialized = True
        else:
            # Monotonic advance while past next_release
            while self.idx + 1 < len(release) and query_np >= next_release[self.idx]:
                self.idx += 1
                self.current_value = values[self.idx]

            # Still not inside current interval?
            if query_np < release[self.idx] or not (query_np < next_release[self.idx]):
                i = find_interval_index(release, next_release, query_np)
                if i < 0:
                    self.initialized = False
                    return None, None
                self.idx = i
                self.current_value = values[i]
                self.initialized = True

        # Valid membership (same rule as find_interval_index)
        if (
            self.idx < len(release)
            and query_np >= release[self.idx]
            and query_np < next_release[self.idx]
        ):
            release_val = release[self.idx]
            age_seconds = (
                query_dt_local - pd.Timestamp(release_val)
            ).total_seconds()
            age_minutes = age_seconds / 60.0
            decay = compute_decay(age_minutes, timeframe)
            return float(self.current_value), decay

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
        features_catalog: Optional[Dict[str, Any]] = None,
        features_to_include: Optional[List[str]] = None,
    ):
        logger.info(f"Loading FSV value intervals from {csv_file}...")
        logger.info(f"  Cache size: {cache_size}")

        self.csv_file = csv_file
        self.intervals_fingerprint = file_fingerprint(csv_file)
        # Active catalog (core + extras); may already be filter-applied by caller.
        self.features_catalog = (
            dict(features_catalog)
            if features_catalog is not None
            else get_features(None, apply_filters=False)
        )

        self.df = pd.read_csv(csv_file)
        self.df["release_datetime"] = pd.to_datetime(self.df["release_datetime"])
        self.df["next_release_datetime"] = pd.to_datetime(
            self.df["next_release_datetime"]
        )

        # Repair any legacy null next_release (open-ended last interval)
        null_next = self.df["next_release_datetime"].isna()
        if null_next.any():
            n = int(null_next.sum())
            logger.warning(
                f"  Repairing {n} null next_release_datetime → open-ended sentinel"
            )
            self.df.loc[null_next, "next_release_datetime"] = pd.Timestamp(
                "2262-01-01 00:00:00"
            )

        self.df = self.df.sort_values(
            ["feature", "release_datetime"]
        ).reset_index(drop=True)

        if features_to_exclude is None:
            features_to_exclude = []
        self.features_to_exclude = list(features_to_exclude)
        self.features_to_include = (
            list(features_to_include) if features_to_include else None
        )

        self._timezone_map = self._build_timezone_map()
        self._index = self._build_interval_index(self.features_to_exclude)

        # Stable order: catalog order, then any leftover CSV-only names
        self.features = self._stable_feature_order(list(self._index.keys()))

        self._state = FeatureState()
        self._get_value_cached = lru_cache(maxsize=cache_size)(self._get_value_impl)

        logger.info(
            f"Loaded {len(self.df):,} records for {len(self._index)} features"
        )
        logger.info(f"Features to exclude: {len(self.features_to_exclude)}")
        if self.features_to_exclude:
            logger.info(f"  Excluded: {self.features_to_exclude}")
        if self.features_to_include is not None:
            logger.info(f"Features whitelist: {self.features_to_include}")

    def _stable_feature_order(self, available: List[str]) -> List[str]:
        avail = set(available)
        catalog_names = list(self.features_catalog.keys())
        ordered = [f for f in catalog_names if f in avail]
        if self.features_to_include is not None:
            include = set(self.features_to_include)
            ordered = [f for f in ordered if f in include]
        # CSV-only names not in catalog: keep only if no whitelist, sorted
        if self.features_to_include is None:
            extras = sorted(f for f in available if f not in set(catalog_names))
            ordered = ordered + extras
        return ordered

    def _build_timezone_map(self) -> Dict[str, str]:
        timezone_map: Dict[str, str] = {}
        for feature, meta in self.features_catalog.items():
            tz = meta.get("timezone", "America/New_York")
            timezone_map[feature] = tz
        return timezone_map

    def _build_interval_index(
        self, features_to_exclude: List[str]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        exclude_set = set(features_to_exclude)
        include_set = (
            set(self.features_to_include)
            if self.features_to_include is not None
            else None
        )
        index: Dict[str, Dict[str, np.ndarray]] = {}

        for feature in self.df["feature"].unique():
            if feature in exclude_set:
                continue
            if include_set is not None and feature not in include_set:
                continue

            mask = self.df["feature"] == feature
            feature_df = self.df.loc[
                mask, ["release_datetime", "next_release_datetime", "value"]
            ].sort_values("release_datetime")

            release = feature_df["release_datetime"].values.astype("datetime64[ns]")
            next_release = feature_df["next_release_datetime"].values.astype(
                "datetime64[ns]"
            )
            # NaT → open-ended sentinel
            nat_mask = np.isnat(next_release)
            if nat_mask.any():
                next_release = next_release.copy()
                next_release[nat_mask] = OPEN_ENDED_NEXT_RELEASE

            index[feature] = {
                "release": release,
                "next_release": next_release,
                "value": feature_df["value"].values.astype("float64"),
                "timeframe": get_timeframe_minutes(feature, self.features_catalog),
            }

        return index

    def _convert_to_local_timezone(
        self,
        query_dt: pd.Timestamp,
        feature: str,
    ) -> pd.Timestamp:
        """Convert UTC datetime to feature's local timezone (naive local wall time)."""
        tz_str = self._timezone_map.get(feature, "America/New_York")

        if query_dt.tz is None:
            query_dt = query_dt.tz_localize("UTC")
        elif str(query_dt.tz) != "UTC":
            query_dt = query_dt.tz_convert("UTC")

        tz = pytz.timezone(tz_str)
        return query_dt.tz_convert(tz).replace(tzinfo=None)

    def _prepare_query_dt_local(
        self,
        query_dt: pd.Timestamp,
        feature: str,
        is_utc: bool = False,
    ) -> pd.Timestamp:
        if is_utc:
            return self._convert_to_local_timezone(query_dt, feature)
        return query_dt

    def _get_value_impl(
        self,
        feature: str,
        query_dt_str: str,
        is_utc: bool = False,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Single-query path — same membership rule as FeatureState."""
        query_dt = pd.Timestamp(query_dt_str)
        query_dt_local = self._prepare_query_dt_local(query_dt, feature, is_utc)

        if feature not in self._index:
            return None, None

        intervals = self._index[feature]
        release_dt = intervals["release"]
        next_release_dt = intervals["next_release"]
        values = intervals["value"]
        timeframe = intervals["timeframe"]

        query_np = np.datetime64(pd.Timestamp(query_dt_local).to_datetime64())
        i = find_interval_index(release_dt, next_release_dt, query_np)
        if i < 0:
            return None, None

        age_seconds = (
            query_dt_local - pd.Timestamp(release_dt[i])
        ).total_seconds()
        age_minutes = age_seconds / 60.0
        decay = compute_decay(age_minutes, timeframe)
        return float(values[i]), decay

    def get_value_decay(
        self,
        feature: str,
        query_datetime: Union[str, datetime, pd.Timestamp],
        is_utc: bool = False,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Get (value, decay) for a feature at a specific datetime."""
        dt_str = str(query_datetime)
        return self._get_value_cached(feature, dt_str, is_utc)

    def get_arranged_features(
        self, features: Optional[List[str]] = None
    ) -> List[str]:
        """
        Return features to query in stable order (FEATURES dict order).
        Excluded features are already absent from self.features.
        """
        if features is None:
            return list(self.features)
        wanted = set(features)
        return [f for f in self.features if f in wanted]

    def get_value_decay_batch_accumulated(
        self,
        query_datetime_from: Union[str, datetime, pd.Timestamp],
        query_datetime_to: Union[str, datetime, pd.Timestamp],
        features: Optional[List[str]] = None,
        is_utc: bool = False,
    ) -> Tuple[List[List[Tuple[Optional[float], Optional[float]]]], Dict[str, int]]:
        """
        Returns results organized by feature:
            [ [ (value, decay) for minutes ] for features ]
        and miss counts per feature.

        Time range is half-open: [from, to) in minutes.
        """
        features = self.get_arranged_features(features)

        dt_from = pd.Timestamp(query_datetime_from)
        dt_to = pd.Timestamp(query_datetime_to)
        total_minutes = int((dt_to - dt_from).total_seconds() // 60)

        if total_minutes <= 0:
            return [], {feature: 0 for feature in features}

        logger.info(
            f"Getting values for {len(features)} features over {total_minutes} minutes"
        )

        # Minute grid once (timestamps, not strings — avoids parse thrash)
        minute_dts = [dt_from + timedelta(minutes=i) for i in range(total_minutes)]

        results_by_feature = []
        miss_counts = {feature: 0 for feature in features}
        state = self._state

        for fid, feature in enumerate(features):
            if feature not in self._index:
                results_by_feature.append([(None, None)] * total_minutes)
                miss_counts[feature] += total_minutes
                continue

            print(
                f"{fid + 1}. Processing feature: {feature} "
                f"({len(minute_dts)} minutes) ..."
            )

            state.reset()
            state.set_intervals(self._index[feature])

            feature_results = []
            tz_str = self._timezone_map.get(feature, "America/New_York")
            tz = pytz.timezone(tz_str) if is_utc else None

            for query_dt in minute_dts:
                if is_utc:
                    # query_dt is naive UTC wall time
                    query_dt_local = (
                        query_dt.tz_localize("UTC")
                        .tz_convert(tz)
                        .replace(tzinfo=None)
                    )
                else:
                    query_dt_local = query_dt

                query_np = np.datetime64(pd.Timestamp(query_dt_local).to_datetime64())
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
        return (
            self.df["release_datetime"].min(),
            self.df["release_datetime"].max(),
        )

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

    print("=" * 60)
    print("FSV Value Lookup - With Cache")
    print("=" * 60)

    data_dir = config.get("data_dir", "dataset_cache")
    os.makedirs(data_dir, exist_ok=True)

    features_to_exclude = config.get("features_to_exclude", [])
    features_to_include = config.get("features_to_include")
    # Full meta for decay/tz (extras included); include/exclude filter at index time.
    catalog = get_features(config, apply_filters=False)
    intervals_csv = f"{data_dir}/fsv_value_intervals.csv"

    lookup = FSVValueLookup(
        intervals_csv,
        features_to_exclude=features_to_exclude,
        features_to_include=features_to_include,
        features_catalog=catalog,
        cache_size=10000,
    )

    print(f"\nExcluded features: {lookup.get_excluded_features()}")
    print(f"Available features: {len(lookup.get_features())}")
    print(f"Intervals fingerprint: {lookup.intervals_fingerprint}")

    dt_from = config.get("binance_start_date", "2024-01-01 00:00:00")
    dt_to = config.get("binance_end_date", "2026-07-01 00:00:00")
    features = lookup.get_arranged_features(None)
    is_utc = True

    print(f"\nFeatures to query: {features}, len={len(features)}")
    print(f"Time range: {dt_from} to {dt_to}")
    print(f"Is UTC: {is_utc}")

    cache_filename = generate_cache_filename(
        features=features,
        query_datetime_from=dt_from,
        query_datetime_to=dt_to,
        is_utc=is_utc,
        features_to_exclude=features_to_exclude,
        intervals_fingerprint=lookup.intervals_fingerprint,
    )
    cache_path = os.path.join(data_dir, cache_filename)

    print(f"\nCache file: {cache_path}")

    arr = np.array([], dtype=np.float32)
    total_minutes = 0
    miss_counts = {}

    cache_valid = False
    if os.path.exists(cache_path):
        print("\nCache hit! Loading from file...")
        start = time.time()
        arr = np.load(cache_path)
        elapsed = time.time() - start
        print(f"Loaded array shape: {arr.shape}")
        print(f"Load time: {elapsed:.4f}s")

        metadata_path = cache_path.replace(".npy", "_meta.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            # Validate cache identity
            if metadata.get("intervals_fingerprint") != lookup.intervals_fingerprint:
                print(
                    "⚠ Cache intervals fingerprint mismatch — recomputing..."
                )
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
                arr = np.array([], dtype=np.float32)
            elif metadata.get("features") != features:
                print("⚠ Cache feature list mismatch — recomputing...")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
                arr = np.array([], dtype=np.float32)
            else:
                miss_counts = metadata.get("miss_counts", {})
                total_minutes = metadata.get(
                    "total_minutes",
                    arr.shape[1] if len(arr.shape) > 1 else 0,
                )
                cache_valid = True
        else:
            # No metadata → accept array but prefer recompute next time if unsure
            miss_counts = {}
            total_minutes = arr.shape[1] if len(arr.shape) > 1 else 0
            cache_valid = True

    if not cache_valid:
        print("\nCache miss! Computing...")

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
            print(
                f"Results shape: {total_features} features x {total_minutes} minutes"
            )
            print(f"Compute time: {compute_time:.4f}s")

            print("\nConverting to numpy array...")
            start = time.time()
            arr = np.array(results, dtype=np.float32)
            convert_time = time.time() - start

            print(f"Array shape: {arr.shape}")
            print(f"Array dtype: {arr.dtype}")
            print(f"Convert time: {convert_time:.4f}s")

            print(f"\nSaving to cache: {cache_path}")
            np.save(cache_path, arr)

            metadata = {
                "features": features,
                "total_minutes": total_minutes,
                "miss_counts": miss_counts,
                "dt_from": str(dt_from),
                "dt_to": str(dt_to),
                "is_utc": is_utc,
                "shape": list(arr.shape),
                "features_to_exclude": lookup.get_excluded_features(),
                "intervals_fingerprint": lookup.intervals_fingerprint,
                "intervals_csv": intervals_csv,
            }
            metadata_path = cache_path.replace(".npy", "_meta.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved metadata to: {metadata_path}")

    # ==========================================================
    # Null/NaN Value Analysis
    # ==========================================================

    print("\n" + "=" * 60)
    print("Null/NaN Value Analysis")
    print("=" * 60)

    if arr.size == 0:
        print("\nArray is empty. No analysis to perform.")
        print("Please use a valid time range (dt_to > dt_from).")
    else:
        nan_mask = np.isnan(arr)
        nan_count = nan_mask.sum()
        total_elements = arr.size

        print(f"\nTotal elements in array: {total_elements:,}")
        print(
            f"Total NaN values: {nan_count:,} "
            f"({nan_count / total_elements * 100:.2f}%)"
        )

        print("\n" + "-" * 60)
        print("Feature-level Null Analysis")
        print("-" * 60)

        feature_null_stats = []
        for i, feature in enumerate(features):
            if arr.ndim < 3:
                break
            feature_nan_count = nan_mask[i, :, :].sum()
            feature_nan_pct = feature_nan_count / (arr.shape[1] * 2) * 100

            miss_count = miss_counts.get(feature, 0)
            miss_pct = (
                miss_count / total_minutes * 100 if total_minutes > 0 else 0
            )

            feature_null_stats.append(
                {
                    "feature": feature,
                    "nan_count": feature_nan_count,
                    "nan_pct": feature_nan_pct,
                    "miss_count": miss_count,
                    "miss_pct": miss_pct,
                    "total_minutes": total_minutes,
                }
            )

        feature_null_stats.sort(key=lambda x: x["nan_pct"], reverse=True)

        print(
            f"\n{'Feature':<30} {'NaN Count':<12} {'NaN %':<10} "
            f"{'Miss Count':<12} {'Miss %':<10}"
        )
        print("-" * 75)

        for stats in feature_null_stats:
            if stats["nan_count"] > 0 or stats["miss_count"] > 0:
                print(
                    f"{stats['feature']:<30} {stats['nan_count']:<12} "
                    f"{stats['nan_pct']:<10.2f} {stats['miss_count']:<12} "
                    f"{stats['miss_pct']:<10.2f}"
                )

        fully_null_features = [
            s for s in feature_null_stats if s["nan_pct"] >= 100
        ]
        if fully_null_features:
            print("\n" + "-" * 60)
            print("Features with 100% null values (completely missing):")
            for s in fully_null_features:
                print(f"  - {s['feature']}")

        partially_null = [
            s for s in feature_null_stats if 0 < s["nan_pct"] < 100
        ]
        if partially_null:
            print("\n" + "-" * 60)
            print("Features with partial null values:")
            for s in partially_null[:10]:
                print(f"  - {s['feature']}: {s['nan_pct']:.2f}% null")
            if len(partially_null) > 10:
                print(
                    f"  ... and {len(partially_null) - 10} more features "
                    f"with partial nulls"
                )

        no_null_features = [
            s for s in feature_null_stats if s["nan_pct"] == 0
        ]
        if no_null_features:
            print("\n" + "-" * 60)
            print(
                f"Features with no null values: "
                f"{len(no_null_features)} out of {len(features)}"
            )
            print(
                f"  {', '.join([s['feature'] for s in no_null_features[:10]])}"
            )
            if len(no_null_features) > 10:
                print(f"  ... and {len(no_null_features) - 10} more")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    # Always return (n_channels, T) float32:
    #   n_channels = 2 * num_features  (value, decay per feature)
    # Empty catalog / failed range → (0, T) so callers can use shape[1].
    if arr.size == 0:
        try:
            t0 = pd.Timestamp(dt_from)
            t1 = pd.Timestamp(dt_to)
            T_empty = max(0, int((t1 - t0).total_seconds() // 60))
        except Exception:
            T_empty = 0
        empty = np.zeros((0, T_empty), dtype=np.float32)
        print(f"⚠ Returning empty macro array with shape {empty.shape}")
        return empty, list(lookup.features)

    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(
            f"Expected FSV array shape (n_features, T, 2); got {arr.shape}. "
            f"Delete the FSV cache under data_dir and recompute."
        )

    shape = arr.shape  # (num_features, num_minutes, 2)
    # (value_0, decay_0, value_1, decay_1, ...) rows × minutes
    reshaped = arr.transpose(0, 2, 1).reshape(-1, shape[1])

    return reshaped, list(lookup.features)


if __name__ == "__main__":

    from config import get_config

    config = get_config()
    arr, macro_feature_str_list = get_value_age_array(config)
