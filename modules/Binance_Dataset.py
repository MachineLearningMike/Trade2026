import os
import sys
import numpy as np
from datetime import datetime, timedelta, timezone

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.Legacy.Binance_new import Binance

class Binance_Dataset:
    def __init__(self, save_dir):
        super().__init__()
        
        self.save_dir = save_dir
        
        # mark = dt.month, dt.day, dt.weekday, dt.hour, dt.minute
        self.mark_mean = np.array([6.5, 16, 3, 11.5, 29.5])
        self.mark_std = np.array([pow((n-m)*(n-m+2)/12, 0.5) for (m, n) in [(1, 12), (1, 31), (0, 6), (0, 23), (0, 59)]])   # population std
        
        self.binance = Binance.instantiate(connect=False)
        
    def get_paths(self, start_utc):
        folder_path = Path(self.save_dir) / str(start_utc.year).zfill(4) / str(start_utc.month).zfill(2) / str(start_utc.day).zfill(2)
        date_anchor = datetime(start_utc.year, start_utc.month, start_utc.day)
        minutes = int((start_utc - date_anchor).total_seconds() / 60)
        file_path_means = folder_path / (str(minutes).zfill(4) + "_means.npy")    
        file_path_volumes = folder_path / (str(minutes).zfill(4) + "_volumes.npy")
        return folder_path, file_path_means, file_path_volumes
    
    def get_minutes_from_file_name(self, file_name):
        parts = file_name.split('_')
        return int(parts[0]), parts[1]


    def build_binance_dataset(self, symbol, start_utc, end_utc, interval, save_dir, minute_step):

        start_utc = start_utc - timedelta(minutes=minute_step)
        while start_utc < end_utc:

            start_utc = start_utc + timedelta(minutes=minute_step)
            end_utc = start_utc + timedelta(minutes=2*24*60)
            print(f"------------------------------------------------ {start_utc} -----------------------------------")
            
            folder_path, file_path_means, file_path_volumes = self.get_paths(start_utc)
            
            # exist = True
            # if not folder_path.exists(): exist = False
            # if exist: continue
            
            folder_path.mkdir(parents=True, exist_ok=True)

            exist = False
            try:
                # arr = np.load(file_path_means)
                # arr = np.load(file_path_volumes)
                assert file_path_means.exists()
                assert file_path_volumes.exists()
                exist = True
            except: pass
            if exist: continue
                
            candles, nFills, successful, _, reason = self.binance.Get_Price_Data_By_Time("klines", symbol, interval, start_utc, end_utc)
            print(f"{start_utc} {successful}")
            if not successful: continue

            means = (candles[:, 1] + candles[:, 2]) / 2
            volumes = candles[:, 4]

            np.save(file_path_means, means)
            np.save(file_path_volumes, volumes)
    
    def is_consistent(self, arr):
        report = True
        try:
            assert arr.shape[0] == 2880
            assert not np.isnan(arr).any()
            # assert arr[arr < 0.5].shape[0] <= 0
            # assert arr[arr > 2.0].shape[0] <= 0
        except:
            report = False
        return report
     
    def load_datapoint(self, dt, value='plain'):
        _, file_path_means, file_path_volumes = self.get_paths(dt)        

        means, volumes, norm_marks = None, None, None
        try:
            means = np.load(file_path_means)           #------------- ugly but we need an even number for FFT.
            
            if value == 'ratio':
                means = (means[1:]) / (means[:-1] + 1e-9)
            elif value == 'log_ratio':
                means = np.log((means[1:] + 1e-9) / (means[:-1] + 1e-9)) * 1000
            elif value == 'plain':
                means = means
            else: 
                print(f"Invalid value_type: {value}")
                raise Exception('Invalid value')
            
            assert self.is_consistent(means)
              
        except:
            means = None
            return means, volumes, norm_marks
        
        marks = []
        minutes = means.shape[0]
        for m in range(minutes):
            d = dt + timedelta(minutes=m)
            marks.append([d.month, d.day, d.weekday(), d.hour, d.minute])
        marks = np.array(marks)     # (n, 5)
        norm_marks = (marks - self.mark_mean) / self.mark_std
        
        return means, volumes, norm_marks

                
    def get_datapoint_datetimes(self):
    
        base_folder = Path(self.save_dir)
    
        def datetime_iter():
            for year_path in sorted(base_folder.iterdir()):
                if not year_path.is_dir() or not year_path.name.isdigit():
                    continue
                year = int(year_path.name)

                for month_path in sorted(year_path.iterdir()):
                    if not month_path.is_dir() or not month_path.name.isdigit():
                        continue
                    month = int(month_path.name)

                    for day_path in sorted(month_path.iterdir()):
                        if not day_path.is_dir() or not day_path.name.isdigit():
                            continue
                        day = int(day_path.name)

                        for file_path in sorted(day_path.iterdir()):
                            if not file_path.is_file():
                                continue

                            minutes, var_string = self.get_minutes_from_file_name(file_path.stem)

                            yield (datetime(year, month, day) + timedelta(minutes=minutes))

        datetitmes = list(datetime_iter())
        datetitmes = list(set(datetitmes))

        return datetitmes                


    def analyze_datapoints(self, inclusive_left=None, inclusive_right=None, minute_step=10):

        # Step 1 — Scan all hours first
    
        datetimes = self.get_datapoint_datetimes()

        existing_points = sorted(datetimes)
        point_min = existing_points[0]
        point_max = existing_points[-1]

        # Step 2 — Convert boundaries
        left = point_min if inclusive_left is None else inclusive_left
        right = point_max if inclusive_right is None else inclusive_right

        if left > right:
            raise ValueError("inclusive_left must be <= inclusive_right")

        # Step 3 — Keep only hours inside the interval
        trimmed_points = sorted([p for p in datetimes if left <= p and p <= right])

        first_point = trimmed_points[0]
        last_point = trimmed_points[-1]

        # Step 4 — Identify missing hours in the window
        full_range = []
        dt = first_point
        while dt <= last_point:
            full_range.append(dt)
            dt = dt + timedelta(minutes=minute_step)

        missing = set(full_range) - set(trimmed_points)
        unexpected = set(trimmed_points) - set(full_range)

        return {
            "count": len(trimmed_points),
            "first_point": first_point,
            "last_point": last_point,
            "n_missings": len(missing),
            "n_unexpected": len(unexpected),
        }
        

if __name__ == "__main__":

    data_dir = "d:\\trade_data\\history"
    binance = Binance_Dataset(data_dir)
    
    start_utc = datetime(2018, 1, 1)
    end_utc = datetime(2026, 3, 16)
    symbol = 'BTCUSDT'
    interval = '1m'
    minute_step = 10   # minutes

    # binance.build_binance_dataset(symbol, start_utc, end_utc, interval, data_dir, minute_step)

    datetimes = binance.get_datapoint_datetimes()
    
    report = binance.analyze_datapoints(inclusive_left=None, inclusive_right=None, minute_step=10)
    print(report)