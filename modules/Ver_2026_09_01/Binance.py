import time as tm
from datetime import datetime, timedelta
import pytz
from binance.client import Client
import numpy as np
import pandas as pd
import sys
import os
import threading
import urllib.request
import requests
import patoolib
from pathlib import Path

# ==================== Config Setup ====================
try:
    from config import Config
except ImportError:
    Config = {
        'api_key': '',
        'api_secret': '',
        'exchange': 'binance.com',
        'binance_wait_sec': 0.5,
    }

# ==================== STORE_DIRECTORY ====================
# Set default STORE_DIRECTORY if not in environment
if 'STORE_DIRECTORY' not in os.environ:
    os.environ['STORE_DIRECTORY'] = 'D:/trading_data'

STORE_DIRECTORY = os.environ['STORE_DIRECTORY']
print(f"📂 Using STORE_DIRECTORY: {STORE_DIRECTORY}")

# ==================== Print Function ====================
lockPrint = threading.Lock()

def Print(*argv, end="\n"):
    str_builder = ''
    for arg in argv:
        str_builder += str(arg)
    lockPrint.acquire()
    print(str_builder, end=end)
    lockPrint.release()

# ==================== Utility Functions ====================

def intervalToMilliseconds(interval):
    seconds_per_unit = {"m": 60, "h": 60 * 60, "d": 24 * 60 * 60, "w": 7 * 24 * 60 * 60}
    unit = interval[-1]
    if unit in seconds_per_unit:
        try:
            return int(interval[:-1]) * seconds_per_unit[unit] * 1000
        except ValueError:
            pass
    return None

def get_timeslot(dataType):
    return 0 if dataType == 'klines' else None

def get_current_month_start(dt_any):
    return datetime(dt_any.year, dt_any.month, 1).replace(hour=0, minute=0, second=0, microsecond=0).replace(tzinfo=pytz.utc)

def get_next_month_start(dt_any):
    return (dt_any.replace(day=1) + timedelta(days=32)).replace(day=1, hour=0, minute=0, second=0, microsecond=0).replace(tzinfo=pytz.utc)

def get_current_day_start(dt_any):
    return datetime(dt_any.year, dt_any.month, dt_any.day).replace(hour=0, minute=0, second=0, microsecond=0).replace(tzinfo=pytz.utc)

def get_next_day_start(dt_any):
    return (dt_any.replace(hour=0) + timedelta(hours=25)).replace(hour=0, minute=0, second=0, microsecond=0).replace(tzinfo=pytz.utc)

def get_previoius_day_start(dt_any):
    return (dt_any.replace(hour=0) + timedelta(hours=-1)).replace(hour=0, minute=0, second=0, microsecond=0).replace(tzinfo=pytz.utc)

def create_empty_dynarray():
    return np.empty([0, 12], dtype=float)

def parse_datalines_to_datalines(dataType, datalines):
    if dataType == 'klines':
        for idx, dl in enumerate(datalines):
            dl[0] = round(dl[0])
            dl[1] = float(dl[1])
            dl[2] = float(dl[2])
            dl[3] = float(dl[3])
            dl[4] = float(dl[4])
            dl[5] = float(dl[5])
            dl[6] = round(dl[6])
            dl[7] = float(dl[7])
            dl[8] = round(dl[8])
            dl[9] = float(dl[9])
            dl[10] = float(dl[10])
            dl[11] = float(dl[11])
            datalines[idx] = dl
    return datalines

def is_one_incremental(array):
    if array.shape[0] < 2:
        return True
    delta = array[1:] - array[:-1]
    return round(delta.max()) == 1 and round(delta.min()) == 1

def starting_index_of_1st_gap(array):
    if array.shape[0] < 2:
        return None
    delta = array[1:] - array[:-1]
    gap_indices = np.where(delta > 1)[0]
    return int(gap_indices[0]) if len(gap_indices) > 0 else None

def snap_sort_deduplicate_in_opening_time(dynarray, interval_mili):
    dynarray[:, 0] = (dynarray[:, 0] // interval_mili) * interval_mili
    dynarray[:, 6] = (dynarray[:, 6] // interval_mili) * interval_mili
    
    if dynarray.shape[0] >= 2:
        sort = np.argsort(dynarray[:, 0], axis=0)
        dynarray = dynarray[sort]
        delta = dynarray[1:, 0] - dynarray[:-1, 0]
        dynarray = np.insert(dynarray[1:][delta >= interval_mili], 0, dynarray[0], axis=0)
    
    return dynarray

def normalize_timestamps_by_magnitude(dynarray, timestamp_indices=[0, 6]):
    if dynarray.shape[0] == 0:
        return dynarray
    
    MS_LOW, MS_HIGH = 946684800000, 4102444800000
    US_LOW, US_HIGH = MS_LOW * 1000, MS_HIGH * 1000
    NS_LOW, NS_HIGH = MS_LOW * 1_000_000, MS_HIGH * 1_000_000
    
    for col in timestamp_indices:
        if col >= dynarray.shape[1]:
            continue
        
        sample = next((dynarray[i, col] for i in range(min(10, dynarray.shape[0])) 
                      if not np.isnan(dynarray[i, col]) and dynarray[i, col] != 0), None)
        
        if sample is None:
            continue
        
        if MS_LOW <= sample <= MS_HIGH:
            pass
        elif US_LOW <= sample <= US_HIGH:
            dynarray[:, col] = dynarray[:, col] // 1000
        elif NS_LOW <= sample <= NS_HIGH:
            dynarray[:, col] = dynarray[:, col] // 1_000_000
        else:
            dynarray[:, col] = dynarray[:, col].astype(np.int64)
            mask = (dynarray[:, col] >= 0) & (dynarray[:, col] <= 2147483647)
            dynarray[mask, col] = dynarray[mask, col] + 4294967296
            dynarray[:, col] = dynarray[:, col] // 1000
    
    return dynarray

def select_from_dynarray_by_time(dataType, dynarray, start_utc, end_utc):
    timeslot = get_timeslot(dataType)
    start_timestamp = round(datetime.timestamp(start_utc) * 1000)
    
    if end_utc is not None:
        end_timestamp = round(datetime.timestamp(end_utc) * 1000)
        return dynarray[(start_timestamp <= dynarray[:, timeslot]) & (dynarray[:, timeslot] < end_timestamp)]
    else:
        return dynarray[start_timestamp <= dynarray[:, timeslot]]

def read_csv_file(file_path):
    try:
        return pd.read_csv(file_path, header=None, index_col=None, dtype={0: 'int64', 6: 'int64'})
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def get_base_path(datetime_utc, dataType, symbol, interval, dailyNotMonthly=True):
    """Get base path with binance_data folder"""
    if dataType == 'klines':
        if dailyNotMonthly:
            base_path = f"binance_data/spot/daily/klines/{symbol.upper()}/{interval}/"
            file_name = f"{symbol.upper()}-{interval}-{datetime_utc.strftime('%Y-%m-%d')}.zip"
        else:
            base_path = f"binance_data/spot/monthly/klines/{symbol.upper()}/{interval}/"
            file_name = f"{symbol.upper()}-{interval}-{datetime_utc.strftime('%Y-%m')}.zip"
    else:
        raise Exception('Invalid dataType.')
    return base_path, file_name

def get_file_location(datetime_utc, dataType, symbol, interval, dailyNotMonthly=True):
    """Get file location using STORE_DIRECTORY"""
    base_folder, file_name = get_base_path(datetime_utc, dataType, symbol, interval, dailyNotMonthly)
    # Use STORE_DIRECTORY from environment
    save_folder = os.path.join(STORE_DIRECTORY, base_folder)
    save_path = os.path.join(save_folder, file_name)
    return base_folder, file_name, save_folder, save_path

def download_file(current_utc, dataType, symbol, interval, dailyNotMonthly=True):
    BASE_URL_DOWNLOAD = 'https://data.binance.vision/'
    base_folder, file_name, save_folder, save_path = get_file_location(
        current_utc, dataType, symbol, interval, dailyNotMonthly
    )
    download_path = os.path.join(base_folder, file_name)
    csv_path = str(Path(save_path).with_suffix('.csv'))
    
    if os.path.exists(csv_path):
        Print(f"\nFile already exists: {csv_path}")
        return csv_path
    
    os.makedirs(save_folder, exist_ok=True)
    
    try:
        if not os.path.exists(save_path):
            download_url = f"{BASE_URL_DOWNLOAD}{download_path}"
            Print(f"\nDownloading: {download_url}")
            dl_file = urllib.request.urlopen(download_url)
            length = dl_file.getheader('content-length')
            length = int(length) if length else 0
            blocksize = max(4096, length // 100) if length else 4096
            
            with open(save_path, 'wb') as out_file:
                while True:
                    buf = dl_file.read(blocksize)
                    if not buf:
                        break
                    out_file.write(buf)
        
        if not os.path.exists(csv_path):
            patoolib.extract_archive(save_path, outdir=save_folder)
            Print(f"Extracted to: {csv_path}")
    
    except urllib.error.HTTPError as e:
        Print(f"\nFile not found: {download_url} (HTTP {e.code})")
        csv_path = None
    except Exception as e:
        Print(f"\nDownload error: {e}")
        csv_path = None
    
    return csv_path

# ==================== Binance API Call ====================

class ClientPool:
    def __init__(self, total):
        self.total = total
        self.clients = [Client() for _ in range(total)]
        self.clientIdInUse = -1
    
    def Next(self):
        self.clientIdInUse = 0 if self.clientIdInUse >= len(self.clients) - 1 else self.clientIdInUse + 1
        return self.clients[self.clientIdInUse]
    
    def Repair(self, client):
        if client in self.clients:
            idx = self.clients.index(client)
        else:
            idx = self.clientIdInUse
        self.clients[idx] = Client()

def Call_Binance_API_v2(clientPool, method_name, nTrials, wait_sec, *args, **kwargs):
    for _ in range(nTrials):
        try:
            client = clientPool.Next()
            method = getattr(Client, method_name)
            result = method(client, *args, **kwargs)
            return result, round(datetime.timestamp(datetime.utcnow().replace(tzinfo=pytz.utc)) * 1000)
        except:
            tm.sleep(wait_sec)
    return None, None

def exact_fetch(clientPool, binance_wait_sec, dataType, symbol, interval, startTime, endTime_inclusive, nToFetch, interval_mili):
    datalines = []
    lastTime = startTime - interval_mili
    
    while lastTime < endTime_inclusive:
        maxLimit = min(500, nToFetch)
        
        if dataType == 'klines':
            lines, _ = Call_Binance_API_v2(
                clientPool, "get_klines", 1, binance_wait_sec,
                symbol=symbol, interval=interval, limit=maxLimit,
                startTime=lastTime + interval_mili, endTime=endTime_inclusive
            )
        else:
            lines = None
        
        if lines:
            datalines.extend(lines)
            lastTime = lines[-1][0]
        else:
            return None
    
    return datalines

# ==================== Data Fetching Functions ====================

def get_all_from_historic_file(sample_utc, dataType, symbol, interval, dailyNotMonthly=True):
    """Get data from historic file using STORE_DIRECTORY"""
    base_folder, file_name, save_folder, save_path = get_file_location(
        sample_utc, dataType, symbol, interval, dailyNotMonthly
    )
    filepath = str(Path(os.path.join(save_folder, file_name)).with_suffix('.csv'))
    
    if not os.path.exists(filepath):
        path = download_file(sample_utc, dataType, symbol, interval, dailyNotMonthly)
        if not path:
            return None, 0, False
    
    dataframe = read_csv_file(filepath)
    if dataframe is None:
        return None, 0, False
    
    interval_mili = intervalToMilliseconds(interval)
    dynarray = dataframe.to_numpy()
    dynarray[:, 11] = 0
    dynarray = normalize_timestamps_by_magnitude(dynarray, timestamp_indices=[0, 6])
    dynarray = snap_sort_deduplicate_in_opening_time(dynarray, interval_mili)
    
    return dynarray, 0, True

def fill_rightward(clientPool, binance_wait_sec, dynarray, dataType, symbol, interval, search_start_mili, search_end_mili):
    nForged = 0
    interval_mili = intervalToMilliseconds(interval)
    
    if dynarray.shape[0] == 0:
        return dynarray, nForged, True
    
    sort_idx = np.argsort(dynarray[:, 0])
    dynarray = dynarray[sort_idx]
    
    mask = (dynarray[:, 0] >= search_start_mili) & (dynarray[:, 0] < search_end_mili)
    dynarray = dynarray[mask]
    
    search_start_idx = 0
    end_idx_exclusive = dynarray.shape[0]
    
    while search_start_idx < end_idx_exclusive - 1:
        slice_array = dynarray[search_start_idx:end_idx_exclusive, 0] / interval_mili
        gap_start_relative = starting_index_of_1st_gap(slice_array)
        
        if gap_start_relative is None:
            break
        
        gap_start_idx = search_start_idx + gap_start_relative
        assert gap_start_idx + 1 < end_idx_exclusive
        
        start_mili = round(dynarray[gap_start_idx, 0] + interval_mili)
        end_mili_inclusive = round(dynarray[gap_start_idx + 1, 0]) - interval_mili
        nRowsToFill = round((end_mili_inclusive - start_mili) / interval_mili) + 1
        
        # Rightward fill using last known price
        source_price = dynarray[gap_start_idx, 4]
        pArray = np.full(nRowsToFill, source_price, dtype=float)
        
        gap_start_mili = dynarray[gap_start_idx, 0]
        array = np.ndarray(shape=(nRowsToFill, 12), dtype=float)
        array[:, 0] = np.array([gap_start_mili + interval_mili * (i + 1) for i in range(nRowsToFill)], dtype=float)
        array[:, 1] = pArray  # Open
        array[:, 2] = pArray  # High
        array[:, 3] = pArray  # Low
        array[:, 4] = pArray  # Close
        array[:, 5] = 0.0
        array[:, 6] = array[:, 0] + interval_mili
        array[:, 7] = 0.0
        array[:, 8] = 0.0
        array[:, 9] = 0.0
        array[:, 10] = 0.0
        array[:, 11] = -4  # Marker: rightward fill
        
        dynarray = np.insert(dynarray, gap_start_idx + 1, array, axis=0)
        nForged += nRowsToFill
        search_start_idx = gap_start_idx + nRowsToFill + 1
        end_idx_exclusive = dynarray.shape[0]
    
    return dynarray, nForged, True

# ==================== Binance Class (Simplified) ====================

class Binance:
    _singleton = None
    clientPool = None
    
    def __init__(self):
        raise RuntimeError('Call instance() instead')
    
    @classmethod
    def instantiate(cls, connect=True):
        if cls._singleton is None:
            cls._singleton = cls.__new__(cls)
            if connect:
                cls.clientPool = ClientPool(1)
        return cls._singleton
    
    def GetFeatureList(self):
        return ["Open", "High", "Low", "Close", "Volume", "QuoteVolume", 
                "NumTrades", "BuyBaseVolume", "BuyQuoteVolume", "Mark"]
    
    def Get_Price_Data_By_Time(self, dataType, symbol, interval, start_utc, end_utc):
        """Fetch price data from Binance historical files with gap filling"""
        assert interval in ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
        
        if start_utc is None:
            start_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
        start_utc = start_utc.replace(tzinfo=pytz.utc)
        
        if end_utc is None:
            end_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
        end_utc = end_utc.replace(tzinfo=pytz.utc)
        
        interval_mili = intervalToMilliseconds(interval)
        start_mili = round(datetime.timestamp(start_utc) * 1000 / interval_mili) * interval_mili
        end_mili = round(datetime.timestamp(end_utc) * 1000 / interval_mili) * interval_mili
        
        nRowsExpected = int((end_mili - start_mili) / interval_mili)
        if nRowsExpected <= 0:
            return create_empty_dynarray(), 0, True, False, "No data expected"
        
        start_org = start_utc
        end_org = end_utc
        start_utc = start_utc - timedelta(days=2)
        end_utc = end_utc + timedelta(days=2)
        
        dynarray = create_empty_dynarray()
        data_loaded = False
        
        # Try monthly files first
        focus_month_start = get_current_month_start(start_utc)
        end_month_start = get_current_month_start(end_utc)
        
        while focus_month_start <= end_month_start:
            array, fills, success = get_all_from_historic_file(
                focus_month_start, dataType, symbol, interval, dailyNotMonthly=False
            )
            if success and array is not None and array.shape[0] > 0:
                data_loaded = True
                dynarray = np.append(dynarray, array, axis=0)
            focus_month_start = get_next_month_start(focus_month_start)
        
        # If monthly failed, try daily files
        if not data_loaded:
            focus_day_start = get_current_day_start(start_utc)
            end_day_start = get_current_day_start(end_utc)
            
            while focus_day_start <= end_day_start:
                array, fills, success = get_all_from_historic_file(
                    focus_day_start, dataType, symbol, interval, dailyNotMonthly=True
                )
                if success and array is not None and array.shape[0] > 0:
                    data_loaded = True
                    dynarray = np.append(dynarray, array, axis=0)
                focus_day_start = get_next_day_start(focus_day_start)
        
        if not data_loaded:
            return create_empty_dynarray(), 0, False, False, "No data available"
        
        # Fill gaps and filter to requested range
        dynarray = snap_sort_deduplicate_in_opening_time(dynarray, interval_mili)
        dynarray, nForged, _ = fill_rightward(
            self.clientPool, Config['binance_wait_sec'], dynarray, 
            dataType, symbol, interval, start_mili, end_mili
        )
        dynarray = select_from_dynarray_by_time(dataType, dynarray, start_org, end_org)
        
        # Remove timestamp columns
        if dynarray.shape[1] >= 7:
            dynarray = np.delete(dynarray, [0, 6], axis=1)
        dynarray = np.float32(dynarray)
        
        successful = (dynarray.shape[0] == nRowsExpected)
        reason = "" if successful else "Row count mismatch"
        
        return dynarray, nForged, successful, False, reason
    
    def get_combined_history(self, datatype, interval, start, end, symbol_dict):
        features_to_return, nForged = [], 0
        
        for symbol in symbol_dict.keys():
            features = symbol_dict[symbol]
            feature_list = self.GetFeatureList()
            
            candles, nFills, successful, _, reason = self.Get_Price_Data_By_Time(
                datatype, symbol, interval, start, end
            )
            
            if successful:
                print(f"Success! {symbol}: (nCandles, nAttributes) = {candles.shape}, nForged = {nFills}")
                nForged += nFills
                
                for feature in features:
                    features_to_return.append(candles[:, feature_list.index(feature)])
            else:
                print(f"Failed to fetch {symbol}: {reason}")
        
        return features_to_return, nForged

    def get_combined_history_2(self, datatype, interval, start, end, symbol_dict, target):
        features_to_return, nForged, target_feature_id = [], 0, None
        target_symbol, target_feature = target

        feature_names = []
        feature_id = 0
        for symbol in symbol_dict.keys():
            features = symbol_dict[symbol]
            feature_list = self.GetFeatureList()
            
            candles, nFills, successful, _, reason = self.Get_Price_Data_By_Time(datatype, symbol, interval, start, end)
            print(reason if reason else "No reason provided.")
            if successful:
                print("Successful !!!!!!!!!! (nCandles, nAttributes) = {}, nForged = {}".format(candles.shape, nFills))
                nForged += nFills
                
                for feature in features:
                    features_to_return.append(candles[:, feature_list.index(feature)])
                    if symbol == target_symbol and feature == target_feature:
                        target_feature_id = feature_id
                    feature_id += 1

                    feature_names.append(f"{symbol}_{feature}")
        
        return features_to_return, nForged, target_feature_id, feature_names

    def get_combined_history_3(self, datatype, interval, start, end, symbol_dict, target):
        # Keep a list of 1-D series (one array per feature). Callers (data.py)
        # stack themselves; returning an (F, T) ndarray makes `if features`
        # raise "truth value of an array is ambiguous".
        return self.get_combined_history_2(
            datatype, interval, start, end, symbol_dict, target
        )

# ==================== Main Entry Point ====================

if __name__ == "__main__":
    print(f"📂 Data directory: {STORE_DIRECTORY}")
    print(f"📂 Full path: {os.path.join(STORE_DIRECTORY, 'binance_data', 'spot')}")
    
    # Ensure directory exists
    os.makedirs(os.path.join(STORE_DIRECTORY, 'binance_data', 'spot'), exist_ok=True)
    
    bn = Binance.instantiate(connect=False)
    
    datatype = "klines"
    interval = '1m'
    start = datetime(2024, 1, 1)
    end = datetime(2026, 6, 30)
    
    symbol_dict = {
        "ETHUSDT": ["Open", "High", "Low", "Close", "Volume"],
        "BTCUSDT": ["Open", "High", "Low", "Close", "Volume"]
    }
    
    features, nForget = bn.get_combined_history(datatype, interval, start, end, symbol_dict)
    
    T = features[0].shape[0] if len(features) > 0 else 0
    for feature in features:
        assert feature.shape[0] == T, "All features must have the same number of time points."
    
    print(f"nFeatures ========== {len(features)}, T ========== {T}, nForged ========== {nForget}")