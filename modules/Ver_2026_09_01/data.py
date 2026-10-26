# data.py
# Dataset and DataLoader with per-window normalization and caching.
# Task APIs live in task_base / Task_*.py; this module
# owns series fetch, windowing, norm/label caches, and loaders.

from __future__ import annotations

import os
from pathlib import Path

import math
import torch
from torch.utils.data import Dataset, Subset
import threading
import queue
import random
import time
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Any, Mapping, Sequence, Union
import pickle
import hashlib
import json
from config import get_config
from datetime import datetime

from Binance import Binance

from telepred import (
    signature_fields as telepred_signature_fields,
    resolve_delay as telepred_resolve_delay,
    resolve_past_window,
    valid_bot_times,
    windows_for_sample,
)
# Bump when label *implementation* / cache format changes but config params
# stay the same. Used only inside the pickle — never hashed into
# generate_dataset_signature / checkpoint keys.
# v7: past-regime aux removed (future-only).
# v8: Task.build_raw_y owns price windowing + label.
# v9: flat prefixed cache keys x.* / y.* / inter.* / intra.* (task not label).
# v10: inter.normalization_layers (config normalize_window removed).
# v11: inter.stats['x'] = train-only global feature mean/std for one-layer.
# v12: if Task.normalize_y, label_tensor stored train-normalized.

class TradingDataset(Dataset):
    """Trading dataset. Construct with TradingDataset(config, task)."""

    LABEL_CACHE_ALGO_VERSION = 12
    BINANCE_CACHE_VERSION = 1

    @staticmethod
    def generate_dataset_signature(config: dict, task) -> str:
        """Hash label-cache / dataset identity from config + task."""
        symbol_dict = config.get('binance_symbols') or {}
        target = config.get('prediction_target', ('UNKNOWN', 'UNKNOWN'))
        market_feature_names = []
        price_feature_idx = None
        for symbol, features in symbol_dict.items():
            for feature in features:
                if symbol == target[0] and feature == target[1]:
                    price_feature_idx = len(market_feature_names)
                market_feature_names.append(f"{symbol}_{feature}")
        assert price_feature_idx is not None

        from macro_dictionary import get_features
        macro_feature_names = [
            suffix
            for name in get_features(config, apply_filters=True)
            for suffix in (name + "_value", name + "_decay")
        ]
        feature_names = market_feature_names + macro_feature_names

        start = datetime.strptime(
            str(config['binance_start_date']).strip(), '%Y-%m-%d %H:%M:%S'
        )
        end = datetime.strptime(
            str(config['binance_end_date']).strip(), '%Y-%m-%d %H:%M:%S'
        )
        interval = str(config.get('binance_interval', '1m')).strip().lower()
        units = {'m': 1, 'h': 60, 'd': 60 * 24, 'w': 60 * 24 * 7}
        interval_min = (
            int(interval[:-1]) * units[interval[-1]]
            if interval[-1] in units else int(interval)
        )
        T = int((end - start).total_seconds() / 60) // interval_min
        assert T > 0

        key_params = {
            'binance_datatype': config.get('binance_datatype', 'klines'),
            'binance_interval': config.get('binance_interval', '1m'),
            'binance_start_date': config.get('binance_start_date'),
            'binance_end_date': config.get('binance_end_date'),
            'binance_symbols': config.get('binance_symbols'),
            'macros_to_exclude': sorted(config.get('macros_to_exclude', [])),
            'macros_to_include': sorted(config.get('macros_to_include') or []),
            'prediction_target': list(target),
            'price_feature_idx': price_feature_idx,
            'feature_names': list(feature_names),
            'F': len(feature_names),
            'T': T,
            'num_momentum': config.get('num_momentum', 4),
            'past_window': resolve_past_window(config),
            'future_window': int(task.resolve_future_window(config)),
            'step': config.get('step', 1),
            'label': task.name,
            'output_dim': config.get('output_dim', 3),
            'normalization_layers': TradingDataset.resolve_normalization_layers(config),
            'train_ratio': config.get('train_ratio', 0.70),
            'val_ratio': config.get('val_ratio', 0.20),
            'shuffle_split': config.get('shuffle_split', False),
        }
        key_params.update(telepred_signature_fields(config))
        key_params.update(task.signature_fields(config))
        return hashlib.md5(
            json.dumps(key_params, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]


    @staticmethod
    def _save_binance_cache(
        candles_path: Path,
        meta_path: Path,
        key: str,
        key_params: dict,
        features: List[np.ndarray],
        T: int,
        nForget: int,
        target_feature_id: int,
        feature_names: List[str],
    ) -> Path:
        """Persist stacked candles + sidecar meta next to ``candles_<key>.npy``."""
        candles_path.parent.mkdir(parents=True, exist_ok=True)
        features_arr = np.stack(
            [np.asarray(f, dtype=np.float32) for f in features], axis=0
        )
        if features_arr.ndim != 2 or features_arr.shape[1] != int(T):
            raise ValueError(
                f"Binance cache features shape {features_arr.shape} incompatible "
                f"with T={T}"
            )
        tmp_candles = candles_path.with_suffix('.tmp.npy')
        tmp_meta = meta_path.with_suffix('.tmp.json')

        np.save(tmp_candles, features_arr)
        os.replace(tmp_candles, candles_path)

        # Sidecar meta is required: npy alone lacks target_feature_id, names, and
        # key_params for config-consistency checks on load.
        meta = {
            'binance_key': key,
            'cache_version': TradingDataset.BINANCE_CACHE_VERSION,
            'key_params': key_params,
            'T': int(T),
            'F': int(features_arr.shape[0]),
            'nForget': int(nForget) if nForget is not None else 0,
            'target_feature_id': int(target_feature_id),
            'feature_names': list(feature_names),
            'payload': candles_path.name,
            'saved_at': time.time(),
        }
        with open(tmp_meta, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, default=str)
        os.replace(tmp_meta, meta_path)
        return candles_path


    @staticmethod
    def _load_binance_cache(candles_path: Path, meta_path: Path, key: str, key_params: dict):
        """
        Load cached Binance market data from flat keyed files.

        Returns:
            (features_list, T, nForget, target_feature_id, feature_names) or None
        """
        if not meta_path.exists() or not candles_path.exists():
            return None
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            if meta.get('binance_key') not in (None, key):
                print(
                    f"⚠ Binance cache key mismatch "
                    f"(meta={meta.get('binance_key')}, expected={key}); ignoring cache."
                )
                return None
            meta_params = meta.get('key_params')
            if not isinstance(meta_params, dict):
                print(
                    "⚠ Binance cache missing key_params (old schema); "
                    "ignoring cache."
                )
                return None
            if meta.get('cache_version') != TradingDataset.BINANCE_CACHE_VERSION:
                print(
                    f"⚠ Binance cache_version "
                    f"{meta.get('cache_version')!r} != "
                    f"{TradingDataset.BINANCE_CACHE_VERSION}; ignoring cache."
                )
                return None
            expected = json.dumps(key_params, sort_keys=True, default=str)
            actual = json.dumps(meta_params, sort_keys=True, default=str)
            if actual != expected:
                print(
                    "⚠ Binance cache key_params mismatch; ignoring cache.\n"
                    f"  meta: {actual}\n"
                    f"  want: {expected}"
                )
                return None
            features_arr = np.load(candles_path)
            if features_arr.ndim != 2:
                print(
                    f"⚠ Binance cache {candles_path.name} has shape "
                    f"{features_arr.shape}; expected 2-D (F, T)."
                )
                return None
            T_meta = int(meta.get('T', features_arr.shape[1]))
            F_meta = int(meta.get('F', features_arr.shape[0]))
            if features_arr.shape != (F_meta, T_meta):
                print(
                    f"⚠ Binance cache shape {features_arr.shape} != meta "
                    f"(F={F_meta}, T={T_meta}); ignoring cache."
                )
                return None
            feature_names = list(meta.get('feature_names') or [])
            if feature_names and len(feature_names) != features_arr.shape[0]:
                print(
                    f"⚠ Binance cache feature_names length "
                    f"{len(feature_names)} != F={features_arr.shape[0]}; ignoring."
                )
                return None
            if not feature_names:
                feature_names = [f'feature_{i}' for i in range(features_arr.shape[0])]
            nForget = int(meta.get('nForget', 0))
            target_feature_id = int(meta.get('target_feature_id', 0))
            features = [
                features_arr[i].astype(np.float32, copy=False)
                for i in range(features_arr.shape[0])
            ]
            return features, T_meta, nForget, target_feature_id, feature_names
        except Exception as e:
            print(f"⚠ Failed to load Binance cache {candles_path.name}: {e}")
            return None


    @staticmethod
    def get_binance_data(config):
        """Load market series from cache or Binance. Returns features, T, nForget, target_id, names."""
        symbol_dict = config.get('binance_symbols') or {}
        pred = config.get('prediction_target', ('UNKNOWN', 'UNKNOWN'))
        key_params = {
            'cache_version': TradingDataset.BINANCE_CACHE_VERSION,
            'binance_datatype': config.get('binance_datatype', 'klines'),
            'binance_interval': config.get('binance_interval', '1m'),
            'binance_start_date': config.get('binance_start_date'),
            'binance_end_date': config.get('binance_end_date'),
            'binance_symbols': [
                {'symbol': str(sym), 'columns': list(cols)}
                for sym, cols in symbol_dict.items()
            ],
            'prediction_target': list(pred) if pred is not None else None,
        }
        key = hashlib.md5(
            json.dumps(key_params, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        cache_dir = Path(config.get('data_dir', 'dataset_cache')) / 'binance'
        candles_path = cache_dir / f'candles_{key}.npy'
        meta_path = cache_dir / f'candles_{key}_meta.json'

        print(f"🔑 Binance data key: {key}")
        print(f"📁 Binance cache file: {candles_path}")

        cached = TradingDataset._load_binance_cache(
            candles_path, meta_path, key, key_params
        )
        if cached is not None:
            features, T, nForget, target_feature_id, feature_names = cached
            print(f"✓ Binance cache hit ({candles_path.name})")
            print(f"  Number of features: {len(features)}")
            print(f"  Time points: {T}")
            print(f"  Forgotten points: {nForget}")
            print(f"  Target feature Id: {target_feature_id}")
            return features, T, nForget, target_feature_id, feature_names
        print("⚠ Binance cache miss — fetching from source.")

        try:
            bn = Binance.instantiate(connect=False)
            start = datetime.strptime(
                config['binance_start_date'], '%Y-%m-%d %H:%M:%S'
            )
            end = datetime.strptime(
                config['binance_end_date'], '%Y-%m-%d %H:%M:%S'
            )
            print(
                f"Fetching Binance data: {config['binance_datatype']}/"
                f"{config['binance_interval']} {start} → {end} | "
                f"{list(symbol_dict.keys())}"
            )
            features, nForget, target_feature_id, feature_names = (
                bn.get_combined_history_3(
                    config['binance_datatype'],
                    config['binance_interval'],
                    start,
                    end,
                    symbol_dict,
                    pred,
                )
            )
            assert isinstance(features, (list, tuple)) and len(features) > 0
            T = int(features[0].shape[0])
            for feature in features:
                assert feature.shape[0] == T
            print(
                f"Binance data loaded: F={len(features)} T={T} "
                f"nForget={nForget} target={target_feature_id}"
            )
            try:
                saved = TradingDataset._save_binance_cache(
                    candles_path,
                    meta_path,
                    key,
                    key_params,
                    features,
                    T,
                    nForget,
                    target_feature_id,
                    feature_names,
                )
                print(f"✓ Saved Binance data → {saved}")
            except Exception as e:
                print(f"⚠ Failed to save Binance cache: {e}")
            return features, T, nForget, target_feature_id, feature_names
        except Exception as e:
            print(f"⚠ Error fetching Binance data: {e}")
            return None, None, None, None, None


    @staticmethod
    def resolve_normalization_layers(config: Optional[dict] = None) -> str:
        """
        Return config['normalization_layers']: 'two-layer' | 'one-layer'.

        Default: 'two-layer'.
        """
        cfg = config or {}
        raw = cfg.get('normalization_layers', 'two-layer')
        mode = str(raw).strip().lower().replace('_', '-')
        if mode in ('two-layer', 'twolayer', '2-layer', '2layer'):
            return 'two-layer'
        if mode in ('one-layer', 'onelayer', '1-layer', '1layer'):
            return 'one-layer'
        raise ValueError(
            f"normalization_layers must be 'one-layer' or 'two-layer', got {raw!r}"
        )


    @staticmethod
    def _pack_cached_part(
        *,
        mean_tensor: torch.Tensor,
        std_tensor: torch.Tensor,
        momentum_tensor: torch.Tensor,
        label_tensor: torch.Tensor,
        stats: dict,
        dataset: TradingDataset,
        config: dict,
    ) -> dict:
        """
        Flat namespaced label-cache payload:

          x.*      — FeaturePrep moments (mean / std / momentum)
          y.*      — raw task targets
          inter.*  — shared dataset / window identity (task-agnostic)
          intra.*  — task-private meta from Task.cache_intra_fields(config)
        """
        task = dataset.task
        out = {
            'cache_algo_version': TradingDataset.LABEL_CACHE_ALGO_VERSION,
            'x.mean_tensor': mean_tensor,
            'x.std_tensor': std_tensor,
            'x.momentum_tensor': momentum_tensor,
            'y.label_tensor': label_tensor,
            'inter.stats': stats,
            'inter.num_samples': len(dataset),
            'inter.F': dataset.F,
            'inter.T': dataset.T,
            'inter.past_window': dataset.past_window,
            'inter.future_window': dataset.future_window,
            'inter.step': dataset.step,
            'inter.max_accum_deviation': dataset.max_accum_deviation,
            'inter.task': dataset.label_mode,
            'inter.normalization_layers': dataset.normalization_layers,
            'inter.price_feature_idx': dataset.price_feature_idx,
            'inter.cache_key': dataset.cache_key,
            'inter.timestamp': time.time(),
        }
        for key, value in task.cache_intra_fields(config).items():
            out[f'intra.{key}'] = value
        return out

    @staticmethod
    def _unpack_cached_part(cached_data: dict):
        """Read prefixed cache tensors/stats; raises KeyError if required keys missing."""
        return (
            cached_data['y.label_tensor'],
            cached_data['x.mean_tensor'],
            cached_data['x.std_tensor'],
            cached_data['x.momentum_tensor'],
            cached_data['inter.stats'],
        )


    def __init__(self, config, task):
        """Build dataset from config + task (loads series, binds knobs, caches)."""
        self.task = task
        task.sync_output_dim(config)

        market_features, _, _, target_feature_id, market_feature_names = (
            TradingDataset.get_binance_data(config)
        )
        assert market_features is not None and target_feature_id is not None
        market_arr = np.stack(market_features, axis=0).astype(np.float32)
        assert market_arr.ndim == 2 and market_arr.shape[1] > 0

        from macro_03_query_value_age import get_value_age_array
        macro_features, macro_feature_names = get_value_age_array(config)
        macro_features = np.asarray(macro_features, dtype=np.float32)
        assert macro_features.ndim == 2
        assert macro_features.shape[1] == market_arr.shape[1]

        macro_feature_names_full = [
            suffix
            for name in macro_feature_names
            for suffix in (name + "_value", name + "_decay")
        ]
        assert macro_features.shape[0] == len(macro_feature_names_full)

        self.total_features = np.concatenate([market_arr, macro_features], axis=0)
        self.n_market_features = market_arr.shape[0]
        self.n_macro_features = macro_features.shape[0]
        self.feature_names = market_feature_names + macro_feature_names_full
        assert self.total_features.shape[1] >= int(resolve_past_window(config)) + int(
            task.resolve_future_window(config)
        )

        self.config = dict(config)
        self.normalization_layers = TradingDataset.resolve_normalization_layers(
            self.config
        )
        self.normalize_window = self.normalization_layers == "two-layer"
        self.config["normalization_layers"] = self.normalization_layers
        self.label_mode = task.name
        self.config["label"] = self.label_mode
        self.past_window = int(resolve_past_window(self.config))
        self.config["past_window"] = self.past_window
        self.future_window = int(task.resolve_future_window(self.config))
        self.config["future_window"] = self.future_window
        self.telepred_delay = int(telepred_resolve_delay(self.config))
        self.config["telepred_delay"] = self.telepred_delay
        self.price_feature_idx = int(target_feature_id)
        self.config["price_feature_idx"] = self.price_feature_idx
        self.step = int(self.config.get("step", 1))
        self.max_accum_deviation = float(
            self.config.get("max_accum_deviation", 12.0)
        )
        self.num_momentum = int(self.config["num_momentum"])
        self.shuffle_split = bool(self.config.get("shuffle_split", False))
        self.horizon_mode = self.config.get("horizon_mode", "floating")
        train_ratio = self.config.get("train_ratio", 0.70)
        val_ratio = self.config.get("val_ratio", 0.20)
        self.split_ratios = (
            train_ratio, val_ratio, 1.0 - train_ratio - val_ratio,
        )

        bound = task.bind_dataset_outputs(self.config)
        self.yield_target = bound.get("yield_target", "both")
        if bound.get("latent_distribution") is not None:
            self.latent_distribution = bound["latent_distribution"]
        self.output_dim = int(bound["output_dim"])
        self.label_channels = bound["label_channels"]

        data_dir_path = Path(self.config.get("data_dir", "dataset_cache"))
        data_dir_path.mkdir(parents=True, exist_ok=True)

        self.Total = torch.from_numpy(
            np.ascontiguousarray(self.total_features.T)
        ).float()
        self.T, self.F = self.Total.shape
        self.valid_t = valid_bot_times(
            self.T,
            self.past_window,
            self.future_window,
            delay=self.telepred_delay,
            step=self.step,
        )
        self.valid_starts = [
            windows_for_sample(
                t, self.telepred_delay, self.past_window, self.future_window
            )["input"].start
            for t in self.valid_t
        ]
        self.cache_key = TradingDataset.generate_dataset_signature(
            self.config, task
        )
        self.cache_file = data_dir_path / f"labels_{self.cache_key}.pkl"

        (
            self.label_tensor,
            self.mean_tensor,
            self.std_tensor,
            self.momentum_tensor,
            self.stats,
        ) = self._load_or_compute_cached_part()

        (
            self.train_indices,
            self.val_indices,
            self.test_indices,
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = self._create_splits()

        self.label_stat_mean = self.stats["label"]["mean"][self.label_channels]
        self.label_stat_std = self.stats["label"]["std"][self.label_channels]
        # Train-only global feature affine for one-layer FeaturePrep.
        if self.normalization_layers == "one-layer":
            self.x_stat_mean = self.stats["x"]["mean"]
            self.x_stat_std = self.stats["x"]["std"]

        print(f"\nTradingDataset initialized:")
        print(f"  Features: {self.F}, Time steps: {self.T}")
    def _create_splits(self):
        """Return train/val/test index lists and Subset views (no self writes)."""
        num_samples = len(self.valid_starts)
        # valid_starts is already chronological. Shuffle only if requested;
        # default is a time-ordered cut (train → val → test).
        indices = list(range(num_samples))

        if self.shuffle_split:
            random.seed(42)
            random.shuffle(indices)

        train_ratio, val_ratio, _test_ratio = self.split_ratios

        train_end = int(train_ratio * num_samples)
        val_end = train_end + int(val_ratio * num_samples)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        return (
            train_indices,
            val_indices,
            test_indices,
            Subset(self, train_indices),
            Subset(self, val_indices),
            Subset(self, test_indices),
        )
    def _compute_cached_part(self):
        label_list, mean_list, std_list, momentum_list = [], [], [], []
        task = self.task
        total_x = self.Total
        total_p = self.Total[:, self.price_feature_idx]
        config = self.config

        # Splits first so one-layer can accumulate train-only global x stats
        # without holding every window in memory.
        (
            self.train_indices,
            self.val_indices,
            self.test_indices,
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = self._create_splits()
        train_index_set = set(self.train_indices)
        # Online moments over train windows: sum / sumsq of shape (F,).
        x_sum = torch.zeros(self.F, dtype=torch.float64)
        x_sumsq = torch.zeros(self.F, dtype=torch.float64)
        x_count = 0

        for idx in tqdm(range(len(self)), desc="Computing labels and norm params"):
            # TelePred: feature window delayed by d; Y(t) anchored at bot time t.
            t = self.valid_t[idx]
            win = windows_for_sample(
                t, self.telepred_delay, self.past_window, self.future_window
            )
            x = total_x[win['input'], :]

            if self.normalize_window:
                from task_base import normalize_x
                _, mean, std, momentum = normalize_x(x, compute_moments=True)
            else:
                mean = torch.zeros(self.F)
                std = torch.ones(self.F)
                momentum = torch.zeros(self.num_momentum, self.F)
                if idx in train_index_set:
                    x64 = x.double()
                    x_sum += x64.sum(dim=0)
                    x_sumsq += (x64 * x64).sum(dim=0)
                    x_count += int(x64.shape[0])
            mean_list.append(mean)
            std_list.append(std)
            momentum_list.append(momentum)

            label = task.build_raw_y(
                t, total_x, total_p, config, mean=mean, std=std
            )
            label_list.append(label)

        # Unnormalized labels and x's properties of all data samples
        label_tensor = torch.stack(label_list)
        mean_tensor = torch.stack(mean_list)
        std_tensor = torch.stack(std_list)
        momentum_tensor = torch.stack(momentum_list)

        train_indices = self.train_indices
        label_train = label_tensor[train_indices]
        mean_train = mean_tensor[train_indices]
        std_train = std_tensor[train_indices]
        momentum_train = momentum_tensor[train_indices]

        stats = {
            'label': {
                'mean': label_train.mean(dim=0),
                'std': label_train.std(dim=0),
            },
            'mean': {
                'mean': mean_train.mean(dim=0),
                'std': mean_train.std(dim=0),
            },
            'std': {
                'mean': std_train.mean(dim=0),
                'std': std_train.std(dim=0),
            },
            'momentum': {
                'mean': momentum_train.mean(dim=0),
                'std': momentum_train.std(dim=0),
            },
        }
        if not self.normalize_window:
            if x_count <= 1:
                raise RuntimeError(
                    "one-layer FeaturePrep needs >1 train window elements "
                    f"to compute global x mean/std (got x_count={x_count})"
                )
            x_mean = (x_sum / x_count).float()
            x_var = (x_sumsq / x_count).float() - x_mean * x_mean
            x_std = x_var.clamp_min(0.0).sqrt().clamp_min(1e-8)
            stats['x'] = {'mean': x_mean, 'std': x_std}

        # Normalize raw_y in-place into the cache when the Task asks for
        # train-normalized datapoint y (stats above stay in raw units).
        if self.task.normalize_y:
            y_mean = stats['label']['mean']
            y_std = stats['label']['std'].clamp_min(1e-8)
            if label_tensor.shape[-1] != y_mean.shape[-1]:
                raise ValueError(
                    f"label_tensor last dim {tuple(label_tensor.shape)} "
                    f"incompatible with label mean {tuple(y_mean.shape)}"
                )
            label_tensor.sub_(y_mean).div_(y_std + 1e-8)

        return (
            label_tensor, mean_tensor, std_tensor, momentum_tensor, stats,
        )
    def _load_or_compute_cached_part(self):
        # Cache key = dataset_signature (NOT epochs / lr / batch_size).
        print(f"Label cache key: {self.cache_key}")
        print(f"  path: {self.cache_file}")
        if self.cache_file.exists():
            print(f"Loading cached labels from: {self.cache_file}")
            try:
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                # Implementation-only version (not part of dataset_signature).
                # Missing key → treat as v1 (pre-raw-yield / pre-prefix format).
                algo_ver = int(
                    cached_data.get(
                        'cache_algo_version',
                        cached_data.get('label_algo_version', 1),
                    )
                )
                algo_ok = algo_ver == TradingDataset.LABEL_CACHE_ALGO_VERSION
                prefixed = (
                    'inter.num_samples' in cached_data
                    and 'y.label_tensor' in cached_data
                    and 'x.mean_tensor' in cached_data
                )
                shape_ok = prefixed and (
                    cached_data.get('inter.num_samples') == len(self) and
                    cached_data.get('inter.F') == self.F and
                    cached_data.get('inter.T') == self.T and
                    cached_data.get('inter.past_window') == self.past_window and
                    cached_data.get('inter.future_window') == self.future_window
                )

                if shape_ok and algo_ok:
                    (
                        label_tensor, mean_tensor, std_tensor,
                        momentum_tensor, stats,
                    ) = TradingDataset._unpack_cached_part(cached_data)

                    print(f"✓ Cache loaded successfully! ({label_tensor.shape[0]} samples)")
                    return (
                        label_tensor, mean_tensor, std_tensor, momentum_tensor,
                        stats,
                    )

                if not prefixed or not algo_ok:
                    print(
                        f"⚠ Cache format/version "
                        f"(algo={algo_ver}, prefixed={prefixed}) "
                        f"!= current {TradingDataset.LABEL_CACHE_ALGO_VERSION}. "
                        f"Recomputing labels (dataset key unchanged)."
                    )
                else:
                    print("⚠ Cache mismatch (shapes/windows). Recomputing...")
                    print(
                        f"  cached: samples={cached_data.get('inter.num_samples')} "
                        f"F={cached_data.get('inter.F')} "
                        f"T={cached_data.get('inter.T')} "
                        f"past={cached_data.get('inter.past_window')} "
                        f"future={cached_data.get('inter.future_window')}"
                    )
                    print(
                        f"  now:    samples={len(self)} F={self.F} T={self.T} "
                        f"past={self.past_window} future={self.future_window}"
                    )
            except Exception as e:
                print(f"⚠ Failed to load cache: {e}. Recomputing...")
        else:
            print("  (no cache file yet for this dataset signature)")

        print("Computing labels and normalization parameters...")
        (
            label_tensor, mean_tensor, std_tensor, momentum_tensor, stats,
        ) = self._compute_cached_part()

        try:
            cache_data = TradingDataset._pack_cached_part(
                mean_tensor=mean_tensor,
                std_tensor=std_tensor,
                momentum_tensor=momentum_tensor,
                label_tensor=label_tensor,
                stats=stats,
                dataset=self,
                config=self.config,
            )
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"✓ Cache saved to: {self.cache_file}")
            print(f"  Cache size: {self.cache_file.stat().st_size / 1024:.2f} KB")
        except Exception as e:
            print(f"⚠ Failed to save cache: {e}")

        return (
            label_tensor, mean_tensor, std_tensor, momentum_tensor, stats,
        )
    def clear_cached_part(self):
        if self.cache_file.exists():
            self.cache_file.unlink()
            print(f"✓ Cache cleared: {self.cache_file}")
    def clear_all_cached_parts(cls, data_dir: Optional[str] = None):
        if data_dir is None:
            config = get_config()
            data_dir = config.get('data_dir', 'dataset_cache')
        cache_path = Path(data_dir)
        if cache_path.exists():
            import shutil
            shutil.rmtree(cache_path)
            print(f"✓ All caches cleared from: {cache_path}")
        else:
            print(f"✓ No cache directory found at: {cache_path}")
    def __len__(self) -> int:
        return len(self.valid_starts)
    def __getitem__(self, idx: int) -> dict:
        """Opaque datapoint dict — schema owned by ``self.task``."""
        return self.task.build_datapoint(self, idx)


class ThreadedDataLoader:
    """DataLoader with continuous prefetch thread."""

    @staticmethod
    def collate_datapoints(datapoints: List[dict]) -> dict:
        """Stack tensor fields; copy non-tensors from the first datapoint."""
        assert datapoints
        assert all(isinstance(d, Mapping) for d in datapoints)
        out: dict = {}
        for key, first in datapoints[0].items():
            if torch.is_tensor(first):
                out[key] = torch.stack([d[key] for d in datapoints], dim=0)
            else:
                out[key] = first
        return out


    @staticmethod
    def batch_to_device(batch: dict, device, non_blocking: bool = True) -> dict:
        """Move tensor fields of a batch dict to device."""
        assert isinstance(batch, Mapping)
        out: dict = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(device, non_blocking=non_blocking)
            else:
                out[k] = v
        return out


    """DataLoader with continuous prefetch thread (doesn't restart per batch)."""
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.current_epoch_indices = None
        self.current_batch_start = 0
        self.epoch = 0
        
        # Larger queue to buffer batches
        self.prefetch_queue = queue.Queue(maxsize=10)
        self.prefetch_thread = None
        self.running = False
        self.iterator = None
        self.load_start_time = None
    def start_epoch(self, epoch, resume_indices=None, resume_batch_start=0):
        """Start a new epoch."""
        # Stop any existing thread
        self._stop_prefetch()
        
        # Clear the queue
        while not self.prefetch_queue.empty():
            try:
                self.prefetch_queue.get_nowait()
            except:
                break
        
        self.epoch = epoch
        
        if resume_indices is not None:
            self.current_epoch_indices = resume_indices
        else:
            self.current_epoch_indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.seed(42 + epoch)
                random.shuffle(self.current_epoch_indices)
        
        self.current_batch_start = resume_batch_start
        self.running = True
        
        # Create iterator BEFORE starting thread
        self._create_iterator()
        
        # Start prefetch thread
        self._start_prefetch()
    def _create_iterator(self):
        """Create iterator with safety checks."""
        if self.current_epoch_indices is None:
            self.current_epoch_indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.seed(42 + self.epoch)
                random.shuffle(self.current_epoch_indices)
        
        # Safety: ensure batch_start is within bounds
        start = self.current_batch_start * self.batch_size
        if start >= len(self.current_epoch_indices):
            print(f"⚠ Batch start {start} exceeds indices length {len(self.current_epoch_indices)}. Resetting.")
            self.current_batch_start = 0
            start = 0
        
        remaining_indices = self.current_epoch_indices[start:]
        
        if len(remaining_indices) == 0:
            print(f"⚠ No remaining indices. Resetting to full epoch.")
            self.current_batch_start = 0
            remaining_indices = self.current_epoch_indices
        
        self.iterator = self._batch_generator(remaining_indices)
    def _batch_generator(self, indices):
        """Generator that yields batches of collated datapoints."""
        batch_size = self.batch_size

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            datapoints = [self.dataset[idx] for idx in batch_indices]
            yield ThreadedDataLoader.collate_datapoints(datapoints)
    def get_total_batches_in_epoch(self):
        """Get total number of batches in current epoch."""
        if self.current_epoch_indices is None:
            return math.ceil(len(self.dataset) / self.batch_size)
        return max(1, math.ceil(len(self.current_epoch_indices) / self.batch_size))
    def get_state(self):
        """Get current state for resuming."""
        return {
            'epoch_indices': self.current_epoch_indices,
            'batch_start': self.current_batch_start,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'epoch': self.epoch,
        }
    def restore_state(self, state):
        """Restore state from checkpoint."""
        self.current_epoch_indices = state['epoch_indices']
        self.current_batch_start = state['batch_start']
        self.batch_size = state['batch_size']
        self.shuffle = state['shuffle']
        self.epoch = state.get('epoch', 0)
    def _start_prefetch(self):
        """Start the prefetch thread (only if not already running)."""
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            return
        
        self.load_start_time = time.time()
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
    def _prefetch_worker(self):
        """Continuous worker that prefetches ALL batches."""
        try:
            # Safety: ensure iterator exists
            if self.iterator is None:
                print("⚠ Iterator is None in prefetch worker. Creating...")
                self._create_iterator()
            
            while self.running:
                try:
                    # Safety: check if iterator exists
                    if self.iterator is None:
                        print("⚠ Iterator became None. Recreating...")
                        self._create_iterator()
                        if self.iterator is None:
                            self.prefetch_queue.put(('error', RuntimeError("Iterator is None"), 0))
                            return
                    
                    # Metric only batch assembly — NOT queue.put() blocking
                    # (put blocks when the queue is full; that is consumer backpressure,
                    # not data-loading cost, and must not be counted as load_time).
                    t_build0 = time.perf_counter()
                    batch = next(self.iterator)
                    load_time = time.perf_counter() - t_build0
                    self.prefetch_queue.put(('data', batch, load_time))
                except StopIteration:
                    break
                except Exception as e:
                    self.prefetch_queue.put(('error', e, 0))
                    return
            
            if self.running:
                self.prefetch_queue.put(('stop', None, 0))
            self.running = False
            
        except Exception as e:
            self.prefetch_queue.put(('error', e, 0))
    def _stop_prefetch(self):
        """Stop the prefetch thread."""
        self.running = False
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=0.5)
        self.prefetch_thread = None
        
        # Clear the queue
        while not self.prefetch_queue.empty():
            try:
                self.prefetch_queue.get_nowait()
            except:
                break
    def get_next_batch(self):
        """
        Get the next batch from the prefetch queue.

        Blocks until a batch is ready, the epoch ends, or the worker errors.

        Returns:
            (batch, wait_time_s, load_time_s)
            batch is None at end of epoch.
            wait_time_s: time this call blocked on the queue (consumer stall).
            load_time_s: producer-side batch build time only (not put-wait).
        """
        wait_start = time.perf_counter()
        while True:
            try:
                status, data, load_time = self.prefetch_queue.get(timeout=0.5)
            except queue.Empty:
                if self.running or (
                    self.prefetch_thread is not None and self.prefetch_thread.is_alive()
                ):
                    continue
                wait_time = time.perf_counter() - wait_start
                return None, wait_time, 0.0
            except Exception as e:
                print(f"🔍 DEBUG: queue.get() raised: {e}")
                wait_time = time.perf_counter() - wait_start
                return None, wait_time, 0.0

            wait_time = time.perf_counter() - wait_start

            if status == 'stop':
                self.running = False
                return None, wait_time, float(load_time or 0.0)
            if status == 'error':
                raise data

            # Track resume position (batches consumed this epoch)
            self.current_batch_start += 1
            return data, wait_time, float(load_time or 0.0)
    def __del__(self):
        """Cleanup on deletion."""
        self._stop_prefetch()

