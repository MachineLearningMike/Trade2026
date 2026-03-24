import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
import threading
import shutil
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import calendar
import gc
import logging
import math
import platform
import heapq
from multiprocessing import Pool

import json, time, re
from einops import rearrange
from datetime import datetime, date, timedelta, timezone

import random
from torch.utils.data import Dataset
from dateutil.relativedelta import relativedelta
from pathlib import Path

# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import requests

import os, sys, gc
sys.path.append(str(Path(__file__).resolve().parent.parent))
import modules.utils as utils
from modules.Binance_Dataset import Binance_Dataset

class Zeus_Dataset(Dataset):
    def __init__(self, pm):
        super().__init__()
        self.pm = pm
        
        dataset_path = Path(pm.dataset_dir)
        management_path = Path(pm.management_dir)
        
        utils.check_dir(dataset_path)
        utils.check_dir(management_path)
    
        self.param_path = management_path / 'dataset_params.json'
        self.split_points_path = management_path / 'split_points.json'
        
        self.train_points, self.vali_points, self.test_points = None, None, None
        self.train_set, self.vali_set, self.test_set = None, None, None
        self.x_set, self.x_set_len, self.data_slots = None, 0, []

        self._fetched_batches = 0
        self.total_batches = 0
        
        self.origin_date = datetime(1970,1,1)
        
        #-------------------------------------------------------------------------
        self.taken = None
        
        dataset_dir = str(dataset_path)
        self.binance_dataset = Binance_Dataset(dataset_dir)
        
    def hours_since_epoch(self, year, month, day, hour):
        dt = datetime(year, month, day, hour, tzinfo=timezone.utc)
        epoch = datetime(1970, 1, 1, 0, 0, tzinfo=timezone.utc)
        delta = dt - epoch
        return int(delta.total_seconds() // 3600)
    
    def load_data(self):
        # self.mean_meteo, self.std_meteo, self.mean_tempo, self.std_tempo, \
        # self.mean_meteo_ready_cpu, self.std_meteo_ready_cpu, self.mean_meteo_ready_cuda, self.std_meteo_ready_cuda, \
        # self.mean_tempo_ready_cpu, self.std_tempo_ready_cpu, self.mean_tempo_ready_cuda, self.std_tempo_ready_cuda, \
        # self.area_weights_cuda \
        # = self.load_dataset_params(file_name=self.pm.dataset_name, modulo=self.pm.modulo, factor10=self.pm.factor10, is_downsample_and_predict=False)

        self.train_points, self.vali_points, self.test_points = self.split_data_points_by_date_segents(
            self.pm.train_start_dt, self.pm.vali_start_dt, self.pm.test_start_dt, self.pm.test_end_dt_inclusive,
            renew=False, save=False, seed=self.pm.seed
        )
        return
    
    #-------------------------------- data segmentation ---------------------

    def split_data_points_by_date_segents(self, train_start_dt, valid_start_dt, test_start_dt, test_end_dt_inclusive, renew=False, save=True, seed=None):
        """
        Returns:
            train_points, vali_points, test_points : list of datetime objects
        """
        old_train_points = []
        old_vali_points = []
        old_test_points = []

        # Load previous split if exists
        try:
            with open(self.split_points_path, "r") as f:
                data = json.load(f)

            old_train_points = [datetime.fromisoformat(d) for d in data["train_points"]]
            old_vali_points  = [datetime.fromisoformat(d) for d in data["vali_points"]]
            old_test_points  = [datetime.fromisoformat(d) for d in data["test_points"]]
        except:
            print("Loading failed. Creating new data split...")

        # Apply inclusive boundaries to old hours
        def filter_points(points, inclusive_left, exclusive_right):
            filtered = []
            for p in points:
                if inclusive_left is not None and p < inclusive_left:
                    continue
                if exclusive_right is not None and p >= exclusive_right:
                    continue
                filtered.append(p)
            return filtered

        old_train_points = filter_points(old_train_points, train_start_dt, valid_start_dt)
        old_vali_points  = filter_points(old_vali_points, valid_start_dt, test_start_dt)
        old_test_points  = filter_points(old_test_points, test_start_dt, test_end_dt_inclusive + timedelta(milliseconds=1))

        if not renew:
            return old_train_points, old_vali_points, old_test_points   # No mattter whether they are empty, just return the existing status.

        #--- Renew -------------------------------------------
        
        all_points = self.binance_dataset.get_datapoint_datetimes()
        
        def filter_all_points(inclusive_left, exclusive_right):
            points = []
            for dt in all_points:
                if inclusive_left is not None and dt < inclusive_left:
                    continue
                if exclusive_right is not None and dt >= exclusive_right:
                    continue
                points.append(dt)
            return points

        # Inherit old hours that still exist
        train_points_all =  filter_all_points(train_start_dt, valid_start_dt)
        vali_points_all = filter_all_points(valid_start_dt, test_start_dt)
        test_points_all = filter_all_points(test_start_dt, test_end_dt_inclusive + timedelta(hours=1))

        [random.shuffle(train_points_all) for _ in range(10)]
        vali_points_all = sorted(vali_points_all)
        test_points_all = sorted(test_points_all)

        # Save the split as ISO format datetimes
        if save:
            data = {
                "train_points": [p.isoformat() for p in train_points_all],
                "vali_points":  [p.isoformat() for p in vali_points_all],
                "test_points":  [p.isoformat() for p in test_points_all],
            }
            with open(self.split_points_path, "w") as f:
                json.dump(data, f, indent=2)

        return train_points_all, vali_points_all, test_points_all


    def generate_x_points(self, seed=None):
        if self.train_points is None or self.vali_points is None or self.test_points is None:
            print("Call split_data_points_by_date_segents first")
            sys.exit(0)

        # Make copies so the original splits remain unchanged
        train_points = self.train_points[:]
        vali_points  = self.vali_points[:]
        test_points  = self.test_points[:]

        # Shuffle using a local RNG for reproducibility.
        rng = random.Random(seed)
        rng.shuffle(train_points)
        rng.shuffle(vali_points)
        rng.shuffle(test_points)

        return train_points, vali_points, test_points
    
    def get_steps_per_epoch(self):
        return math.ceil(len(self.train_points) / self.pm.batch_size)

    def set_data_slots(self, data_slots):
        self.data_slots = data_slots

    def refresh_x_set(self, x_label, seed=None, renew_if_new_epoch=False):
        renewed = False
        if self.x_set is None or len(self.x_set) <= 0:  # this is a brandnew epoch
            # No need to copy.copy(), as self.x_sets are re-generated above.
            if renew_if_new_epoch is True:
                self.train_points, self.vali_points, self.test_points = self.generate_x_points(seed=seed)

                if platform.system() == "Windows":
                    self.x_set = self.train_points[:] if x_label=='train_set' \
                        else self.vali_points[:] if x_label=='vali_set' \
                        else self.test_points[:] if x_label=='test_set' else None
                else:
                    self.x_set = self.train_points if x_label=='train_set' \
                        else self.vali_points if x_label=='vali_set' \
                        else self.test_points if x_label=='test_set' else None
                
                if x_label == 'train_set': [random.shuffle(self.x_set) for _ in range(10)]
                renewed = True
        
        self._fetched_batches = 0
        self.total_batches = math.ceil(len(self.x_set) / self.pm.batch_size)
        
        return renewed
    
    #-------------------------- pack_state / unpack_state -----------------------------
        
    def pack_state(self):
        """Pack the current x_set state for checkpointing."""
        # We assume the take_next_batch thread has already taken some from self.x_set that's not processed yet.
        if self.taken is not None:
            x_set = list(set(self.taken + self.x_set))      # return back to self.x_set
        else:
            x_set = self.x_set
        return {
            "x_set": [dt.isoformat() for dt in x_set],  # full datetime
            "_fetched_batches": self._fetched_batches,
            "total_batches": self.total_batches,
        }

    def unpack_state(self, data):
        """Restore x_set state from checkpoint data."""    
        self.x_set = [datetime.fromisoformat(s) for s in data["x_set"]]
        self._fetched_batches = 0
        # Recompute total_batches in case batch_size changed
        self.total_batches = math.ceil(len(self.x_set) / self.pm.batch_size)
        
    def no_more_data(self):
        return len(self.x_set) <= 0
    
    #-------------------------------- take_next_batch ---------------------------------
    
    def load_datapoint(self, dt):
        means, volumes, norm_marks = self.binance_dataset.load_datapoint(dt)
        return means, volumes, norm_marks
        
    # def take_next_batch(self):        
    #     if len(self.x_set) <= 0: return None

    #     taken = self.x_set[:self.pm.batch_size]
    #     self.taken = taken
    #     self.x_set = self.x_set[self.pm.batch_size:]
    #     assert len(taken) > 0

    #     batch = []

    #     for b in range(len(taken)):
    #         dt = taken[b]

    #         futures = [
    #             self._disk_threadPool.submit(
    #                 # The reason this range of dt exist in the dataset is that taken[b] exists in the training dataset, which is chronically followed by validation dataset.
    #                     self.load_datapoint, dt
    #                 ) for idx in range(self.pm.nSubstepsPerStep)
    #             ]

    #         for fu in futures:
    #             means, volumes = fu.result()
    #             if means is not None: #  and volumes is not None:
    #                 batch.append(means)
        
    #     return batch
    
    def take_next_batch(self):
        if len(self.x_set) <= 0: 
            return None

        taken = self.x_set[:self.pm.batch_size]
        self.taken = taken
        self.x_set = self.x_set[self.pm.batch_size:]

        # Use list comprehension with single future per datapoint
        futures = [
            self._disk_threadPool.submit(self.load_datapoint, dt) 
            for dt in taken  # One future per datapoint, not per substep
        ]
        
        means, marks = [], []
        for fu in futures:
            _means, _volumes, _norm_marks = fu.result()
            if _means is not None and _norm_marks is not None:
                means.append(_means)
                marks.append(_norm_marks)

        means = torch.from_numpy(np.stack(means, axis=0))
        marks = torch.from_numpy(np.stack(marks, axis=0))
        
        return means, marks
    
    def take_singleton_batch(self, dt):
        _means, _volumes, _norm_marks = self.load_data(dt)
        
        singleton_batch = None
 
        if _means is not None and _norm_marks is not None:
            means, marks = [means], [_norm_marks]
            
            means = torch.from_numpy(np.stack(means, axis=0))
            marks = torch.from_numpy(np.stack(marks, axis=0))
    
            singleton_batch = (means, marks)
 
        return singleton_batch
        
    
    # def load_datapoint_batch(self, dt_list):
    #     batch_data = []
        
    #     # Option 1: Sequential loading within the thread (still parallel across batches)
    #     for dt in dt_list:
    #         means, volumes = self.binance_dataset.load_datapoint(dt)
    #         if means is not None:
    #             batch_data.append(means)
        
    #     return batch_data
    
    # def take_next_batch(self):
    #     if len(self.x_set) <= 0: 
    #         return None

    #     taken = self.x_set[:self.pm.batch_size]
    #     self.taken = taken
    #     self.x_set = self.x_set[self.pm.batch_size:]

    #     # Submit a single task for the entire batch
    #     future = self._disk_threadPool.submit(self.load_datapoint_batch, taken)
        
    #     # Wait for the batch to load
    #     batch = future.result()
        
    #     return batch
    
    #-------------------------- play_physical_prediction -----------------------
    
    def play_physical_prediction(self, inclusive_left, inclusive_right, modulo):
            
        from modules.utils import show_two_arrays, initialize_for_show_array, finalize_for_show_array
    
        def get_key():
            """
            Get the most recent key press without buffering.
            Returns None if no key is currently pressed.
            """
            import keyboard
            import time
            
            # Check all keys we care about
            keys_to_check = {
                'up': 'up', 'down': 'down', 'left': 'left', 'right': 'right',
                '1': '1', '2': '2', '3': '3', '4': '4', '5': '5',
                '6': 'dec', '7': 'mar', '8': 'jun', '9': 'sep',
                'esc': 'quit',
            }
            
            # Check if any key is currently pressed
            for key, action in keys_to_check.items():
                if keyboard.is_pressed(key):
                    # Wait for key release to ensure we don't get multiple triggers
                    while keyboard.is_pressed(key):
                        time.sleep(0.01)
                    return action
            
            return None
        

        initialize_for_show_array()
        
        
        # Main loop -----------------------------------------------------------------------------
        var_id = 0
        dt = inclusive_left
                
        old_path = None; key_stroked = True
        while inclusive_left <= dt and dt <= inclusive_right:
            ss_x, ss_y = None, None
            if key_stroked:
                file_path = self.get_file_path_npz(dt.year, dt.month, dt.day, dt.hour, modulo)
                if file_path != old_path:
                    # dict_data = load_and_decode(file_path)
                    old_path = file_path

                variable = self.pm.meteo_variables[var_id]
                # arr_y = dict_data[variable]  # (C, H, W)

                pred = None; e=s=time.time()
                dt_x = dt - timedelta(hours=modulo)
                if inclusive_left <= dt_x and dt_x <= inclusive_right:
                    pass
                    # meteo_raw_x = torch.from_numpy(self.load_decode_get_array_for_variables(dt_x, modulo=modulo, variables=self.pm.meteo_variables))
                    # ss_x = Snapshot(dt_x.year, dt_x.month, dt_x.day, dt_x.hour, meteo_normal=None, meteo_raw=meteo_raw_x)
                    # ss_y = Snapshot(dt.year, dt.month, dt.day, dt.hour, meteo_normal=None, meteo_raw=None)
                
                    # pred = self.fun_dict[variable].predict(ss_x.meteo_raw, ss_x.dense, ss_x.cat, ss_x.tempo, ss_y.tempo, dt_x, delta_points=modulo).cpu().numpy()
                    # pred = ss_x.meteo_normal[0].numpy() works   # array_x[0].numpy works
            
            # pred = pred # ss_x.meteo_raw[var_id].cpu().numpy() if ss_x is not None else None
            
            # show_two_arrays(arr1=arr_y, arr2=pred, variable1=variable + ".truth", variable2=variable + ".prediction", dt=dt, modulo=modulo)
                        
            # Wait for key input
            key = None
            start_time = time.time()
            while key is None and (time.time() - start_time < 1.0):
                key = get_key()
                if key is None:
                    time.sleep(0.1)  # Small sleep to prevent CPU overuse
            
            # Process key
            if key is None:
                key_stroked = False
                continue
            else:
                 key_stroked = True
            
            if key == "up": 
                var_id += 1
                var_id = 0 if var_id >= len(self.pm.meteo_variables) else var_id
            elif key == "down": 
                var_id -= 1
                var_id = len(self.pm.meteo_variables) - 1 if var_id < 0 else var_id
            elif key == "left": 
                dt = dt - timedelta(hours=modulo)
            elif key == "right": 
                dt = dt + timedelta(hours=modulo)
            elif key == '1':
                modulo = 1
            elif key == '2':
                modulo = 6
            elif key == '3':
                modulo = 12
            elif key == '4':
                modulo = 24
            elif key == '5':
                modulo = 24 * 30
            elif key == 'dec':
                # dict = Snapshot.get_season_events(dt.year)
                dt = dict['dec_solstice'].replace(hour=dt.hour, minute=0)
            elif key == 'mar':
                h = dt.hour
                # dict = Snapshot.get_season_events(dt.year)
                dt = dict['mar_equinox'].replace(hour=dt.hour, minute=0)
            elif key == 'jun':
                h = dt.hour
                # dict = Snapshot.get_season_events(dt.year)
                dt = dict['jun_solstice'].replace(hour=dt.hour, minute=0)
            elif key == 'sep':
                h = dt.hour
                # dict = Snapshot.get_season_events(dt.year)
                dt = dict['sep_equinox'].replace(hour=dt.hour, minute=0)
            elif key == "quit":
                print("Quitting...")
                break
            elif key is None:
                # No key pressed, continue to next iteration
                key_stroked = False
                continue

        finalize_for_show_array()
        
        
    #---------------------------------- Prefetch Loop -------------------------------------
    
    def start_prefetch_thread(self):
        # self.total_batches = None
        # self._fetched_batches = None

        self._lock = threading.Lock()
        self._batch = None
        self._param = None

        self._request_event = threading.Event()
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()

        self._worker_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._worker_thread.start()
        
        self._disk_threadPool = ThreadPoolExecutor(max_workers=4)
        # self._disk_threadPool = Pool(processes=4)
        
        return

    def _prefetch_loop(self):
        while not self._stop_event.is_set():
            # print('prefetch loop... ')
            self._request_event.wait()
            self._request_event.clear()
            if self._stop_event.is_set():
                break

            with self._lock:
                param = self._param

            # (B, T1FG, C, H, W), (B, T2FG, C, H, W), (B, 1, C, H, W), (C,)
            # batch_x, batch_y, batch_diff, zeus_weights
            batch = self.take_next_batch()
            gc.collect()

            with self._lock:
                self._batch = batch
                self._ready_event.set()
            # print(" ...prefetch loop", batch is not None)

    def __prefetch__(self, param):
        with self._lock:
            self._param = param
        self._ready_event.clear()
        self._request_event.set()

    def get_batch(self):    # If __prefetch__ doesn't procedes, it returns all the same batch.
        self._ready_event.wait()
        with self._lock:
            return self._batch

    def __getitem__(self, idx):
        if idx >= self.total_batches:
            raise IndexError("Index exceeds dataset length.")

        batch = self.get_batch()
        if batch is not None:
            self._fetched_batches += 1
        return batch

    def __len__(self):
        return self.total_batches

    def shutdown(self):
        # print('shutdown... ')
        self._stop_event.set()
        self._request_event.set()
        self._worker_thread.join()
        
        self._disk_threadPool.shutdown(wait=True)
        # print(' ...shutdown')
        
    #-----------------------------------------------------------------------------
    
if __name__ == "__main__":
    
    from models.config import params
    
    pm = params
    dataset = Zeus_Dataset(pm)

    dataset.split_data_points_by_date_segents(pm.train_start_dt, pm.vali_start_dt, pm.test_start_dt, pm.test_end_dt_inclusive, renew=True, save=True, seed=None)
    
    # dataset.load_data()