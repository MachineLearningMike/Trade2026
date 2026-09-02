"""
fsv_01_downloader.py

Generic Feature State Vector downloader.

Currently supports:
    - FRED
    - TwelveData

The downloader itself knows nothing about individual features.
Everything is driven by macro_dictionary.py.

For FRED series with use_vintage_data=True, also downloads ALFRED
first-publication dates as ``{feature}__first_release`` columns so
fsv_02 can use real release timing instead of lag heuristics.
"""

import os
from datetime import datetime

import pandas as pd

from macro_dictionary import FEATURES, get_features
from macro_client_fred import FREDClient
from macro_client_twelvedata import TwelveDataClient


class FSVDownloader:

    def __init__(
        self,
        fred_api_key=None,
        twelvedata_api_key=None,
        features_catalog=None,
    ):
        self.clients = {}
        # Full merged catalog (no exclude filters) unless caller passes one.
        self.features_catalog = (
            dict(features_catalog)
            if features_catalog is not None
            else get_features(None, apply_filters=False)
        )

        if fred_api_key is not None:
            self.clients["FRED"] = FREDClient(fred_api_key)

        if twelvedata_api_key is not None:
            self.clients["TwelveData"] = TwelveDataClient(twelvedata_api_key)

    ####################################################################
    # Download one feature
    ####################################################################

    def download_feature(
        self,
        feature_name,
        meta,
        start_date="1990-01-01",
        end_date=None,
    ):
        source = meta["source"]

        if source not in self.clients:
            raise RuntimeError(f"No client configured for {source}")

        client = self.clients[source]
        print(f"Downloading {feature_name}")

        if source == "FRED":
            include_first = bool(meta.get("use_vintage_data", False))
            if include_first:
                print(f"  (fetching ALFRED first-release dates)")
            return client.download_series(
                feature_name=feature_name,
                series_id=meta["series"],
                start_date=start_date,
                end_date=end_date,
                include_first_release=include_first,
            )

        if source == "TwelveData":
            return client.download_symbol(
                feature_name=feature_name,
                symbol=meta["symbol"],
                interval=meta.get("interval", "1day"),
                start_date=start_date,
                end_date=end_date,
            )

        raise RuntimeError(f"Unknown source {source}")

    ####################################################################
    # Download everything
    ####################################################################

    @staticmethod
    def _feature_already_present(feature_name: str, existing_columns) -> bool:
        """True if value column (and vintage companion, if any) already exist."""
        cols = set(existing_columns or [])
        return feature_name in cols

    def download_all(
        self,
        start_date="1990-01-01",
        end_date=None,
        features_catalog=None,
        existing_df=None,
        skip_existing=True,
    ):
        """
        Download catalog features and outer-merge on ``date``.

        If ``existing_df`` is provided (e.g. current ``macro_raw.csv``) and
        ``skip_existing`` is True (default), features whose value column is
        already present are not re-fetched from the API.
        """
        catalog = (
            dict(features_catalog)
            if features_catalog is not None
            else self.features_catalog
        )
        existing = None
        if existing_df is not None and len(existing_df) > 0:
            existing = existing_df.copy()
            if "date" not in existing.columns:
                raise RuntimeError("existing_df must contain a 'date' column")
            existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
            # Parse any __first_release columns as dates
            for c in existing.columns:
                if c.endswith("__first_release"):
                    existing[c] = pd.to_datetime(existing[c], errors="coerce")

        existing_cols = set(existing.columns) if existing is not None else set()
        dfs = []
        failed = []  # list[(name, error)]
        skipped_existing = []
        downloaded = []

        print(f"FSV catalog: {len(catalog)} features")
        if skip_existing and existing_cols:
            print(
                f"Existing raw columns: {len(existing_cols)} "
                f"(will skip features already present)"
            )

        for feature, meta in catalog.items():
            if skip_existing and self._feature_already_present(feature, existing_cols):
                skipped_existing.append(feature)
                print(f"  ↷ skip {feature} (already in macro_raw)")
                continue
            try:
                df = self.download_feature(
                    feature,
                    meta,
                    start_date,
                    end_date,
                )
                # Integrity on value column
                if feature not in df.columns:
                    raise RuntimeError(f"Missing value column for {feature}")
                if df["date"].duplicated().any():
                    df = df.drop_duplicates(subset=["date"], keep="last")
                dfs.append(df)
                downloaded.append(feature)
                print(f"  ✓ {feature}: {len(df):,} rows")
            except Exception as e:
                failed.append((feature, str(e)))
                print(f"  ✗ FAILED {feature}: {e}")

        if skipped_existing:
            print(f"\n↷ Skipped {len(skipped_existing)} existing feature(s)")

        if failed:
            print(f"\n{'=' * 60}")
            print(f"⚠ FAILED DOWNLOADS: {len(failed)} feature(s)")
            print(f"{'=' * 60}")
            print("Names:")
            for name, _err in failed:
                print(f"  - {name}")
            print("Details:")
            for name, err in failed:
                print(f"  - {name}: {err}")
            print(f"{'=' * 60}")

        # Stash for callers (e.g. __main__ writing a failed list file).
        self.last_failed = list(failed)
        self.last_downloaded = list(downloaded)
        self.last_skipped_existing = list(skipped_existing)

        if len(dfs) == 0 and existing is None:
            names = [n for n, _ in failed]
            raise RuntimeError(
                "No features downloaded. Failed: "
                + (", ".join(names) if names else "(none)")
            )

        if len(dfs) == 0:
            # Nothing new — return existing as-is
            merged = existing.sort_values("date").reset_index(drop=True)
            print("\nNo new downloads; keeping existing macro_raw columns:")
            print(merged.columns.tolist())
            return merged

        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on="date", how="outer")

        if existing is not None:
            # Prefer newly downloaded columns on name clash.
            overlap = (set(merged.columns) & set(existing.columns)) - {"date"}
            if overlap:
                existing = existing.drop(columns=list(overlap))
            merged = existing.merge(merged, on="date", how="outer")

        merged = merged.sort_values("date").reset_index(drop=True)

        # Catalog features still missing after merge (failed or absent).
        missing = [f for f in catalog if f not in set(merged.columns)]
        if missing:
            print(f"\n⚠ Catalog features still missing from result ({len(missing)}):")
            for name in missing:
                print(f"  - {name}")

        print("\nResult columns:")
        print(merged.columns.tolist())

        return merged


########################################################################
# Report helpers
########################################################################

def _resolve_data_dir(config=None):
    """Absolute cache dir: env MACRO_DATA_DIR / FSV_DATA_DIR, else project-root data_dir."""
    from config import resolve_data_dir
    if config is not None and config.get("data_dir") is not None:
        return resolve_data_dir(config.get("data_dir"))
    return resolve_data_dir(None)


def _load_macro_raw_report(path: str) -> dict:
    """
    Load prior failed/missing map from ``macro_raw_report.txt``.

    Lines are ``name<TAB>status_or_error`` (``#`` comments / blanks ignored).
    """
    out = {}
    if not os.path.exists(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "\t" in line:
                    name, err = line.split("\t", 1)
                else:
                    name, err = line, ""
                name = name.strip()
                if name:
                    out[name] = err.strip()
    except OSError as e:
        print(f"WARNING: could not read {path}: {e}")
    return out


def _update_macro_raw_report(
    report_path: str,
    dictionary_names,
    present_columns,
    this_run_failed,
    this_run_downloaded=None,
):
    """
    Maintain ``macro_raw_report.txt``.

    Entries are features that either:
      - **failed** to download this run (error text kept), or
      - are **missing**: listed in ``macro_dictionary.FEATURES`` but not present
        as a column in ``macro_raw.csv``.

    Features that appear in the CSV (or succeeded this run) are removed.
    """
    dictionary = list(dictionary_names or [])
    dictionary_set = set(dictionary)
    # CSV columns are the ground truth for "found in the .csv file".
    csv_columns = set(present_columns or [])
    # Treat a successful fetch this run as present even before the CSV rewrite.
    available = csv_columns | set(this_run_downloaded or [])
    prior = _load_macro_raw_report(report_path)

    maintained = {}

    # Prior notes still relevant (in dictionary, still absent from CSV).
    for name, err in prior.items():
        if name in dictionary_set and name not in available:
            maintained[name] = err or "missing from csv (in dictionary)"

    # This-run download failures (may include config extras outside FEATURES).
    for name, err in this_run_failed or []:
        if name in available:
            continue
        maintained[name] = (err or "").strip() or "download failed"

    # Dictionary features not found in the CSV and not already recorded.
    for name in dictionary:
        if name in available:
            continue
        if name not in maintained:
            maintained[name] = "missing from csv (in dictionary)"

    names_sorted = sorted(maintained.keys())
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# FSV raw download report\n")
        f.write("# failed  = download error this run (or still absent after a prior fail)\n")
        f.write(
            "# missing = listed in macro_dictionary.FEATURES but not a column "
            "in macro_raw.csv\n"
        )
        f.write(f"# Updated {datetime.utcnow().isoformat()}Z\n")
        f.write(f"# Count: {len(names_sorted)}\n")
        f.write("# Format: feature_name<TAB>last_error_or_status\n")
        for name in names_sorted:
            f.write(f"{name}\t{maintained[name]}\n")

    if names_sorted:
        print(
            f"\nWARNING: macro_raw_report.txt: "
            f"{len(names_sorted)} failed/missing feature(s)"
        )
        print(f"   -> {report_path}")
        for name in names_sorted:
            print(f"  - {name}: {maintained[name]}")
    else:
        print(
            f"\nOK: macro_raw_report.txt: no failed/missing features "
            f"-> {report_path}"
        )

    return maintained


########################################################################
# Main
########################################################################

def _resolve_api_key(env_name: str, config_key: str, config: dict):
    """Prefer env var, then config. No silent hardcoded fallback."""
    return os.environ.get(env_name) or config.get(config_key) or None


if __name__ == "__main__":

    from config import get_config

    config = get_config()
    # Download core + config extras (do not apply model exclude list here).
    catalog = get_features(config, apply_filters=False)
    print(f"FSV catalog: {len(catalog)} features "
          f"(core + macro_features_extra)")

    # Resolution order: environment → config.py
    FRED_API_KEY = _resolve_api_key("FRED_API_KEY", "fred_api_key", config)
    TWELVEDATA_API_KEY = _resolve_api_key(
        "TWELVEDATA_API_KEY", "twelvedata_api_key", config
    )

    if not FRED_API_KEY:
        raise RuntimeError(
            "FRED API key required. Set env FRED_API_KEY or config['fred_api_key'].\n"
            "PowerShell:  $env:FRED_API_KEY = 'your_key_here'"
        )
    if not TWELVEDATA_API_KEY:
        print(
            "WARNING: TwelveData API key missing "
            "(env TWELVEDATA_API_KEY or config['twelvedata_api_key']). "
            "TwelveData features will fail until set."
        )

    downloader = FSVDownloader(
        fred_api_key=FRED_API_KEY,
        twelvedata_api_key=TWELVEDATA_API_KEY,
        features_catalog=catalog,
    )

    data_dir = _resolve_data_dir(config)
    os.makedirs(data_dir, exist_ok=True)
    raw_path = f"{data_dir}/macro_raw.csv"
    report_path = f"{data_dir}/macro_raw_report.txt"

    existing = None
    if os.path.exists(raw_path):
        print(f"\nLoading existing {raw_path} (skip features already present)...")
        existing = pd.read_csv(raw_path, parse_dates=["date"])
        print(f"  {len(existing):,} rows, {len(existing.columns)} columns")

    fsv = None
    try:
        fsv = downloader.download_all(
            start_date="2020-01-01",
            features_catalog=catalog,
            existing_df=existing,
            skip_existing=True,
        )
    finally:
        # Always refresh the failed/missing list, even when download_all raises.
        present_cols = set(fsv.columns) if fsv is not None else (
            set(existing.columns) if existing is not None else set()
        )
        _update_macro_raw_report(
            report_path=report_path,
            dictionary_names=list(FEATURES.keys()),
            present_columns=present_cols,
            this_run_failed=getattr(downloader, "last_failed", []) or [],
            this_run_downloaded=getattr(downloader, "last_downloaded", []) or [],
        )

    print(fsv.head())

    fsv.to_csv(
        raw_path,
        index=False,
        float_format="%.8g",
    )
    print(f"Saved → {raw_path}")

    # Re-write after save so present columns match the on-disk CSV exactly.
    _update_macro_raw_report(
        report_path=report_path,
        dictionary_names=list(FEATURES.keys()),
        present_columns=set(fsv.columns),
        this_run_failed=getattr(downloader, "last_failed", []) or [],
        this_run_downloaded=getattr(downloader, "last_downloaded", []) or [],
    )
