import time as tm
from datetime import datetime, timedelta
import pytz
from binance.client import Client
import math
import numpy as np
import pandas as pd
import json
import copy
import sys
import os
import threading
import re
import shutil
import urllib.request
import requests
import patoolib
from pathlib import Path
from json import JSONEncoder
from argparse import ArgumentParser, RawTextHelpFormatter, ArgumentTypeError

# ==================== Config Setup ====================
# Try to import config, but provide defaults if not available
try:
    from config import Config
except ImportError:
    # Default configuration
    Config = {
        'api_key': '',
        'api_secret': '',
        'exchange': 'binance.com',
        'Run_Demisser': True,
        'Stream_Price': True,
        'Show_Transceiver': False,
        'Show_Demisser': False,
        'Show_Traders': False,
        'binance_wait_sec': 0.5,
        'max_subscriptions_per_stream': 100,
        'max_slots_per_stream': 100,
        'Bias_Sec_per_Demiss': 0.1,
        'Linear_Sec_per_Lag_X_#gstream': 0.01,
        'start_time_utc': datetime.utcnow().replace(tzinfo=pytz.utc),
        'pathMarketLookup': 'market_lookup.json',
        'STORE_DIRECTORY': os.environ.get('STORE_DIRECTORY', './data')
    }

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
    """Convert a Binance interval string to milliseconds"""
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
    return ms

def get_timeslot(dataType):
    return 0 if dataType == 'klines' else 3 if dataType == 'trades' else 5 if dataType == 'aggTrades' else None

def from_dt_local_to_dt_utc(dt_local):
    now_timestamp = tm.time()
    offset = datetime.fromtimestamp(now_timestamp) - datetime.utcfromtimestamp(now_timestamp)
    dt_utc = dt_local - offset
    dt_utc = dt_utc.replace(tzinfo=pytz.utc)
    return dt_utc

def from_dt_utc_to_dt_local(dt_utc):
    now_timestamp = tm.time()
    offset = datetime.fromtimestamp(now_timestamp) - datetime.utcfromtimestamp(now_timestamp)
    dt_local = dt_utc + offset
    dt_local = dt_local.replace(tzinfo=None)
    return dt_local

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

def get_local_utc_offset_microseconds():
    local_offset_seconds = -time.timezone
    return local_offset_seconds * 1_000_000

def create_empty_dynarray():
    return np.empty([0, 12], dtype=float)

def parse_datalines_to_datalines(dataType, datalines):
    if dataType == 'klines':
        length = len(datalines)
        for idx in range(length):
            dl = datalines[idx]
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
    if len(gap_indices) == 0:
        return None
    return int(gap_indices[0])

def snap_sort_deduplicate_in_opening_time(dynarray, interval_mili):
    org = dynarray.shape[0]
    
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
    
    OVERFLOW_NEG_LOW, OVERFLOW_NEG_HIGH = -2147483648, -1
    OVERFLOW_POS_LOW, OVERFLOW_POS_HIGH = 0, 2147483647
    
    for col in timestamp_indices:
        if col >= dynarray.shape[1]:
            continue
        
        sample = None
        for i in range(min(10, dynarray.shape[0])):
            val = dynarray[i, col]
            if not np.isnan(val) and val != 0:
                sample = val
                break
        
        if sample is None:
            continue
        
        if MS_LOW <= sample <= MS_HIGH:
            pass
        elif US_LOW <= sample <= US_HIGH:
            dynarray[:, col] = dynarray[:, col] // 1000
        elif NS_LOW <= sample <= NS_HIGH:
            dynarray[:, col] = dynarray[:, col] // 1_000_000
        elif (OVERFLOW_NEG_LOW <= sample <= OVERFLOW_NEG_HIGH) or \
             (OVERFLOW_POS_LOW <= sample <= OVERFLOW_POS_HIGH):
            dynarray[:, col] = dynarray[:, col].astype(np.int64)
            recovered = dynarray[:, col].copy()
            mask = (recovered >= 0) & (recovered <= 2147483647)
            recovered[mask] = recovered[mask] + 4294967296
            dynarray[:, col] = recovered // 1000
        else:
            print(f"Warning: Column {col} timestamp {sample:.0f} is outside expected ranges.")
            continue
    
    return dynarray

def select_from_dynarray_by_time(dataType, dynarray, start_utc, end_utc):
    timeslot = get_timeslot(dataType)
    start_timestamp = round(datetime.timestamp(start_utc) * 1000)
    
    if end_utc is not None:
        end_timestamp = round(datetime.timestamp(end_utc) * 1000)
        array = dynarray[np.logical_and(start_timestamp <= dynarray[:, timeslot], dynarray[:, timeslot] < end_timestamp)]
    else:
        array = dynarray[(start_timestamp <= dynarray[:, timeslot])]
    
    return array

def read_csv_file(dataType, file_path):
    dataframe = None
    try:
        dataframe = pd.read_csv(
            file_path,
            header=None,
            index_col=None,
            dtype={0: 'int64', 6: 'int64'}
        )
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        dataframe = None
    return dataframe

def get_base_path(datetime_utc, dataType, symbol, interval, dailyNotMonthly=True):
    if dataType == 'klines':
        if dailyNotMonthly:
            base_path = "data/spot/daily/klines/{}/{}/".format(symbol.upper(), interval)
            file_name = "{}-{}-{}.zip".format(symbol.upper(), interval, datetime_utc.strftime("%Y-%m-%d"))
        else:
            base_path = "data/spot/monthly/klines/{}/{}/".format(symbol.upper(), interval)
            file_name = "{}-{}-{}.zip".format(symbol.upper(), interval, datetime_utc.strftime("%Y-%m"))
    elif dataType == 'trades':
        base_path = "data/spot/daily/trades/{}/".format(symbol.upper())
        file_name = "{}-trades-{}.zip".format(symbol.upper(), datetime_utc.strftime("%Y-%m-%d"))
    elif dataType == 'aggTrades':
        base_path = "data/spot/daily/aggTrades/{}/".format(symbol.upper())
        file_name = "{}-aggTrades-{}.zip".format(symbol.upper(), datetime_utc.strftime("%Y-%m-%d"))
    else:
        raise Exception('Invalid dataType.')
    
    return base_path, file_name

def get_file_location(datetime_utc, dataType, symbol, interval, dailyNotMonthly=True):
    base_folder, file_name = get_base_path(datetime_utc, dataType, symbol, interval, dailyNotMonthly=dailyNotMonthly)
    save_folder = os.path.join(os.environ.get('STORE_DIRECTORY', './data'), base_folder)
    save_path = os.path.join(save_folder, file_name)
    return base_folder, file_name, save_folder, save_path

def download_file(current_utc, dataType, symbol, interval, dailyNotMonthly=True):
    BASE_URL_DOWNLOAD = 'https://data.binance.vision/'
    base_folder, file_name, save_folder, save_path = get_file_location(current_utc, dataType, symbol, interval, dailyNotMonthly=dailyNotMonthly)
    download_path = os.path.join(base_folder, file_name)
    
    csv_path = str(Path(save_path).with_suffix('.csv'))
    if os.path.exists(csv_path):
        Print("\nfile already exists! {}".format(csv_path))
        return csv_path
    
    if not os.path.exists(save_folder):
        Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    try:
        if not os.path.exists(save_path):
            download_url = "{}{}".format(BASE_URL_DOWNLOAD, download_path)
            dl_file = urllib.request.urlopen(download_url)
            length = dl_file.getheader('content-length')
            if length:
                length = int(length)
                blocksize = max(4096, length // 100)
            
            with open(save_path, 'wb') as out_file:
                dl_progress = 0
                Print("\nFile Download: {}".format(save_path))
                while True:
                    buf = dl_file.read(blocksize)
                    if not buf:
                        break
                    dl_progress += len(buf)
                    out_file.write(buf)
                    sys.stdout.flush()
        
        if not os.path.exists(csv_path):
            try:
                patoolib.extract_archive(save_path, outdir=save_folder)
            except:
                os.remove(save_path)
                csv_path = None
    
    except urllib.error.HTTPError:
        Print("\nFile not found: {}".format(download_url), end="")
        csv_path = None
    
    return csv_path

# ==================== Binance API Call ====================

def Call_Binance_API_v2(clientPool, method_name, nTrials, wait_sec, *args, **kwargs):
    result = None
    successful = False
    
    while not successful and nTrials > 0:
        client = None
        try:
            now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
            ts_mili = round(datetime.timestamp(now_utc) * 1000)
            client = clientPool.Next()
            method = getattr(Client, method_name)
            result = method(client, *args, **kwargs)
            successful = True
        except:
            print("Oops!", sys.exc_info()[0], "occurred.")
            clientPool.Repair(client)
            tm.sleep(wait_sec)
            nTrials -= 1
            if nTrials > 0:
                continue
    
    return result, ts_mili

def exact_fetch(clientPool, binance_wait_sec, dataType, symbol, interval, startTime, endTime_inclusive, nToFetch, interval_mili):
    datalines = []
    lastTime = startTime - interval_mili
    
    while lastTime < endTime_inclusive:
        maxLimit = min(500, nToFetch)
        
        if dataType == 'klines':
            nTrials = 1
            lines, ts_mili = Call_Binance_API_v2(
                clientPool, "get_klines", nTrials, binance_wait_sec,
                symbol=symbol, interval=interval, limit=maxLimit,
                startTime=lastTime + interval_mili, endTime=endTime_inclusive
            )
        elif dataType == 'aggTrades':
            lines = None
        else:
            lines = None
        
        if lines is not None and isinstance(lines, list) and len(lines) > 0:
            datalines += lines
            lastTime = lines[-1][0]
        else:
            datalines = None
            break
    
    return datalines

# ==================== Coincodex Functions ====================

ccInterval_sec = 300
ccHistoryFormat = "https://coincodex.com/api/coincodex/get_coin_history/{}/{}/{}/{}"

def fetch_coincodex_history_numpy(symbol, start, end, clientInterval_sec):
    result = None
    assert start <= end
    
    start_sec = round(datetime.timestamp(start) / clientInterval_sec) * clientInterval_sec
    end_sec = round(datetime.timestamp(end) / clientInterval_sec) * clientInterval_sec
    start_clientCandle = round(start_sec / clientInterval_sec)
    end_clientCandle = round(end_sec / clientInterval_sec)
    start_cc = datetime.fromtimestamp((int(clientInterval_sec * start_clientCandle / ccInterval_sec) + 0) * ccInterval_sec)
    end_cc = datetime.fromtimestamp((int(clientInterval_sec * end_clientCandle / ccInterval_sec) + 1) * ccInterval_sec)
    
    start_date = "-".join([str(start_cc.year % 100), str(start_cc.month).zfill(2), str(start_cc.day).zfill(2)])
    end_date = "-".join([str(end_cc.year % 100), str(end_cc.month).zfill(2), str(end_cc.day).zfill(2)])
    delta = end - start
    samples = int((delta.days + 1) * 86400 / ccInterval_sec + 0.1)
    history_url = ccHistoryFormat.format(symbol, start_date, end_date, samples)
    
    try:
        response = requests.put(history_url)
        response.raise_for_status()
        result = response.json()[symbol]
        result = np.array(result)
    except:
        result = None
    
    if result is not None:
        try:
            assert result.shape[0] > 0
            assert round(result[0, 0]) % (ccInterval_sec) == 0
            if result.shape[0] > 1:
                delta = result[1:, 0] - result[:-1, 0]
                assert np.min(delta) == np.max(delta)
        except:
            result = None
    
    history = None
    if result is not None:
        left_to_start = int(clientInterval_sec * start_clientCandle / ccInterval_sec)
        offset = np.argmax(np.round(result[:, 0]) == left_to_start * ccInterval_sec)
        try:
            assert round(result[offset, 0]) == left_to_start * ccInterval_sec
            history = np.zeros((end_clientCandle - start_clientCandle + 1, 4), dtype=float)
            for i in range(start_clientCandle, end_clientCandle + 1, 1):
                left_to_i = int(clientInterval_sec * i / ccInterval_sec)
                id_left = left_to_i - left_to_start + offset
                client_sec = i * clientInterval_sec
                cc_left_sec = left_to_i * ccInterval_sec
                cc_right_sec = cc_left_sec + ccInterval_sec
                try:
                    assert result[id_left, 0] <= client_sec
                    assert client_sec <= result[id_left + 1, 0]
                    assert round(result[id_left, 0]) == cc_left_sec
                    assert round(result[id_left + 1, 0]) == cc_right_sec
                    
                    history[i - start_clientCandle, 0] = client_sec
                    history[i - start_clientCandle, 1:4] = ((cc_right_sec - client_sec) * result[id_left, 1:4] + (client_sec - cc_left_sec) * result[id_left + 1, 1:4]) / ccInterval_sec
                except:
                    history = None
                    break
            
            if history is not None:
                assert history[0, 0] == start_sec
                assert end_sec == history[-1, 0]
                assert history[0, 0] % clientInterval_sec == 0
                if history.shape[0] > 1:
                    delta = history[1:, 0] - history[:-1, 0]
                    assert delta[0] == clientInterval_sec
                    assert np.min(delta) == np.max(delta)
                assert history.shape[0] == end_clientCandle - start_clientCandle + 1
        except:
            history = None
    
    return history

# ==================== Data Fetching Functions ====================

def get_all_from_historic_file(sample_utc, dataType, symbol, interval, dailyNotMonthly=True):
    base_folder, file_name, _, _ = get_file_location(sample_utc, dataType, symbol, interval, dailyNotMonthly)
    folder = os.path.join(os.environ.get('STORE_DIRECTORY', './data'), base_folder)
    filepath = str(Path(os.path.join(folder, file_name)).with_suffix('.csv'))
    
    dynarray = None
    nForged = 0
    successful = True
    
    interval_mili = intervalToMilliseconds(interval)
    
    if not os.path.exists(filepath):
        path = download_file(sample_utc, dataType, symbol, interval, dailyNotMonthly=dailyNotMonthly)
        if path is None or filepath != path:
            return None, 0, False
    
    if successful:
        dataframe = read_csv_file(dataType, filepath)
        
        if dataframe is not None:
            dynarray = dataframe.to_numpy()
            dynarray[:, 11] = 0
            dynarray = normalize_timestamps_by_magnitude(dynarray, timestamp_indices=[0, 6])
            dynarray = snap_sort_deduplicate_in_opening_time(dynarray, interval_mili)
        else:
            successful = False
        
        nForged = 0
    
    return dynarray, nForged, successful

def fill_rightward(clientPool, binance_wait_sec, dynarray, dataType, symbol, interval, search_start_mili, search_end_mili, ignore_last=True):
    nForged = 0
    successful = True
    interval_mili = intervalToMilliseconds(interval)
    
    if dynarray.shape[0] == 0:
        return dynarray, nForged, successful
    
    sort_idx = np.argsort(dynarray[:, 0])
    dynarray = dynarray[sort_idx]
    
    mask = (dynarray[:, 0] >= search_start_mili) & (dynarray[:, 0] < search_end_mili)
    dynarray = dynarray[mask]
    
    search_start_idx = 0
    end_idx_exclusive = dynarray.shape[0]
    
    while search_start_idx < end_idx_exclusive - 1:
        slice_array = dynarray[search_start_idx:end_idx_exclusive, 0] / interval_mili
        gap_start_relative = starting_index_of_1st_gap(slice_array)
        
        if gap_start_relative is not None:
            gap_start_idx = search_start_idx + gap_start_relative
            assert dynarray[gap_start_idx + 1, 0] - dynarray[gap_start_idx, 0] > interval_mili
            assert gap_start_idx + 1 < end_idx_exclusive
            
            start_mili = round(dynarray[gap_start_idx, 0] + interval_mili)
            end_mili_inclusive = round(dynarray[gap_start_idx + 1, 0]) - interval_mili
            nRowsToFill = round((end_mili_inclusive - start_mili) / interval_mili) + 1
            assert nRowsToFill >= 1
            
            rows = None
            if clientPool is not None:
                rows = exact_fetch(clientPool, binance_wait_sec, dataType, symbol, interval,
                                   start_mili, end_mili_inclusive, nRowsToFill, interval_mili)
                assert rows is None
            
            if rows is not None:
                try:
                    if len(rows) != nRowsToFill:
                        rows = None
                    else:
                        rows = parse_datalines_to_datalines(dataType, rows)
                        array = np.array(rows, dtype=float)
                        array = snap_sort_deduplicate_in_opening_time(array, interval_mili)
                        time_ratios = array[:, 0].astype(np.int64) / interval_mili
                        if not is_one_incremental(time_ratios):
                            rows = None
                        else:
                            Print(f"F{nRowsToFill}")
                except Exception:
                    rows = None
            else:
                start_dt = datetime.fromtimestamp(round(start_mili / 1000)).replace(tzinfo=pytz.utc)
                end_dt = datetime.fromtimestamp(round(end_mili_inclusive / 1000)).replace(tzinfo=pytz.utc)
                history = fetch_coincodex_history_numpy(symbol[:-4], start_dt, end_dt, interval_mili / 1000)
                
                history = None
                
                if history is not None:
                    assert history.shape[0] >= nRowsToFill
                    pArray = history[:nRowsToFill, 1]
                    mark = -2
                    Print(f"C{nRowsToFill} ", end="")
                else:
                    source_price = dynarray[gap_start_idx, 4]
                    pArray = np.full(nRowsToFill, source_price, dtype=float)
                    mark = -4
                    Print(f"R{nRowsToFill} ", end="")
                
                gap_start_mili = dynarray[gap_start_idx, 0]
                
                array = np.ndarray(shape=(nRowsToFill, 12), dtype=float)
                array[:, 0] = np.array([gap_start_mili + interval_mili * (i + 1) for i in range(nRowsToFill)], dtype=float)
                array[:, 1] = pArray
                array[:, 2] = pArray
                array[:, 3] = pArray
                array[:, 4] = pArray
                array[:, 5] = 0.0
                array[:, 6] = array[:, 0] + interval_mili
                array[:, 7] = 0.0
                array[:, 8] = 0.0
                array[:, 9] = 0.0
                array[:, 10] = 0.0
                array[:, 11] = mark
            
            assert array.shape[0] == nRowsToFill
            dynarray = np.insert(dynarray, gap_start_idx + 1, array, axis=0)
            assert is_one_incremental(dynarray[gap_start_idx:gap_start_idx + nRowsToFill + 2, 0] / interval_mili)
            
            nForged += nRowsToFill
            search_start_idx = gap_start_idx + nRowsToFill + 1
            end_idx_exclusive = dynarray.shape[0]
        else:
            break
    
    sort_idx = np.argsort(dynarray[:, 0])
    dynarray = dynarray[sort_idx]
    
    _, unique_idx = np.unique(dynarray[:, 0], return_index=True)
    if len(unique_idx) < dynarray.shape[0]:
        dynarray = dynarray[unique_idx]
    
    mask = (dynarray[:, 0] >= search_start_idx) & (dynarray[:, 0] < search_end_mili)
    dynarray = dynarray[mask]
    
    Print(f"... total {nForged} candles forged.")
    return dynarray, nForged, successful

def create_missing_prices_inland(clientPool, binance_wait_sec, dynarray, dataType, symbol, interval, start_mili, end_mili, ignore_last=True):
    assert dynarray is not None and dynarray.shape[0] > 0
    
    nForged = 0
    successful = True
    
    dynarray, nForged, successful = fill_rightward(
        clientPool, binance_wait_sec, dynarray, dataType, symbol, interval,
        0, dynarray.shape[0], ignore_last=ignore_last
    )
    
    return dynarray, nForged, successful

# ==================== ClientPool Class ====================

class ClientPool:
    def __init__(self, total):
        assert total >= 1
        self.total = total
        self.clients = []
        self.clientIdInUse = -1
        self.lock = threading.Lock()
        for n in range(total):
            self.clients.append(Client())
    
    def Next(self):
        self.clientIdInUse = 0 if self.clientIdInUse >= len(self.clients) - 1 else self.clientIdInUse + 1
        return self.clients[self.clientIdInUse]
    
    def Repair(self, client):
        if client in self.clients:
            idx = self.clients.index(client)
        else:
            idx = self.clientIdInUse
        self.clients[idx] = Client()
        return

# ==================== T_State and gstream ====================

class T_State():
    none = 0
    synced = 1
    syncing = 2
    live = 4
    no_missing = 8
    closed = 16
    changed = 32

class gstream():
    lock = None
    dynarray = None
    filepath = None
    stream_id = None
    tickbuffer = None
    t_state = None
    
    def __init__(self, _lock, _dynarray, _filepath, _stream_id, _tick_buffer, _t_state):
        self.lock = _lock
        self.dynarray = _dynarray
        self.filepath = _filepath
        self.stream_id = _stream_id
        self.tickbuffer = _tick_buffer
        self.t_state = _t_state

# ==================== Binance Class ====================

class Binance():
    _singleton = None
    _step_run = 0
    _running = False
    
    client = None
    clientPool = None
    gstream_transceiver_thread = None
    gstream_demisser_thread = None
    
    stop_Maintain_GStreams = None
    stop_GStream_Transceiver_Thread_Function = None
    stop_GStream_Demisser_Thread_Function = None
    
    stream_lock = None
    gstreams = {}
    lock_gstream = None
    gstreams_ready = False
    gstream_datetime_utc = None
    gstream_lag_sec = 0
    unicorn_sockets_manager = None
    streaming = False
    badkeys = []
    
    add_to_plot = None
    demisser_gstream_heartbeat = None
    transceiver_frame_heartbeat = None
    
    def __init__(self):
        raise RuntimeError('Call instance() instead')
    
    @classmethod
    def instantiate(cls, connect=True):
        if cls._singleton is None:
            Binance.Initialize()
            cls.lock_gstream = threading.Lock()
            cls._singleton = cls.__new__(cls)
            
            if connect:
                client = Client(Config['api_key'], Config['api_secret'])
                cls.client = client
                Binance.clientPool = ClientPool(1)
        
        return cls._singleton
    
    @classmethod
    def Initialize(cls):
        Print("Initializing Binance singleton...")
        cls._singleton = None
        cls._step_run = 0
        cls._running = False
        cls.client = None
        cls.clientPool = None
        cls.unicorn_sockets_manager = None
        cls.gstream_transceiver_thread = None
        cls.gstream_demisser_thread = None
        cls.stop_Maintain_GStreams = None
        cls.stop_GStream_Transceiver_Thread_Function = None
        cls.stop_GStream_Demisser_Thread_Function = None
        cls.stream_lock = threading.Lock()
        cls.gstreams = {}
        cls.lock_gstream = None
        cls.gstreams_ready = False
        cls.gstream_datetime_utc = None
        cls.gstream_lag_sec = 0
        cls.unicorn_sockets_manager = None
        cls.streaming = False
        cls.badkeys = []
        cls.demisser_gstream_heartbeat = None
        cls.transceiver_frame_heartbeat = None
        cls.add_to_plot = None
        return
    
    def GetFeatureList(self):
        return ["Open", "High", "Low", "Close", "Volume", "QuoteVolume", "NumTrades", "BuyBaseVolume", "BuyQuoteVolume", "Mark"]
    
    def Get_Price_Data_By_Time(self, dataType, symbol, interval, start_utc, end_utc, lookup=False):
        global file_not_found, no_line_feed
        file_not_found = False
        assert interval in ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
        
        nForged = 0
        successful = True
        reason = ""
        
        if start_utc is None:
            start_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
        else:
            start_utc = start_utc.replace(tzinfo=pytz.utc)
        start_utc = min(start_utc, datetime.utcnow().replace(tzinfo=pytz.utc))
        
        if end_utc is None:
            end_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
        else:
            end_utc = end_utc.replace(tzinfo=pytz.utc)
        end_utc = min(end_utc, datetime.utcnow().replace(tzinfo=pytz.utc))
        
        interval_mili = intervalToMilliseconds(interval)
        
        start_timestamp = datetime.timestamp(start_utc)
        end_timestamp = datetime.timestamp(end_utc)
        start_mili = round(start_timestamp * 1000 / interval_mili) * interval_mili
        end_mili = round(end_timestamp * 1000 / interval_mili) * interval_mili
        start_utc = datetime.fromtimestamp(start_mili / 1000, tz=pytz.utc)
        end_utc = datetime.fromtimestamp(end_mili / 1000, tz=pytz.utc)
        
        nRowsExpected = int((end_mili - start_mili) / interval_mili)
        if nRowsExpected <= 0:
            return create_empty_dynarray(), 0, True, False, "No data expected"
        
        start_org = start_utc
        end_org = end_utc
        margin = 2
        start_utc = start_utc - timedelta(days=margin)
        end_utc = end_utc + timedelta(days=margin)
        
        key = dataType + "." + symbol + "." + interval
        dynarray = create_empty_dynarray()
        slot = get_timeslot(dataType)
        
        leadMsgShown = False
        leadMsg = "Collecting candles from files ... "
        
        end_month_start = get_current_month_start(end_utc)
        month_loops = 0
        focus_month_start = get_current_month_start(start_utc)
        
        success = False
        data_loaded = False
        
        while focus_month_start <= end_month_start:
            month_loops += 1
            
            if not leadMsgShown:
                Print(leadMsg, end="")
                leadMsgShown = True
            
            array, fills, success = get_all_from_historic_file(focus_month_start, dataType, symbol, interval, dailyNotMonthly=False)
            
            if success and array is not None and array.shape[0] > 0:
                data_loaded = True
                nForged += fills
                dynarray = np.append(dynarray, array, axis=0)
            else:
                break
            
            focus_month_start = get_next_month_start(focus_month_start)
        
        if not success:
            failed_focus_month_start = focus_month_start
            end_day_start = get_current_day_start(end_utc)
            day_loops = 0
            focus_day_start = get_current_day_start(focus_month_start)
            success = False
            while focus_day_start <= end_day_start:
                day_loops += 1
                array, fills, success = get_all_from_historic_file(focus_day_start, dataType, symbol, interval, dailyNotMonthly=True)
                if success and array is not None and array.shape[0] > 0:
                    data_loaded = True
                    nForged += fills
                    dynarray = np.append(dynarray, array, axis=0)
                else:
                    break
                
                focus_day_start = get_next_day_start(focus_day_start)
        
        if data_loaded:
            dynarray = snap_sort_deduplicate_in_opening_time(dynarray, interval_mili)
            dynarray, nForged_fill, fill_success = fill_rightward(
                self.clientPool, Config['binance_wait_sec'], dynarray, dataType, symbol, interval,
                start_mili, end_mili
            )
            nForged += nForged_fill
            dynarray = select_from_dynarray_by_time(dataType, dynarray, start_org, end_org)
            successful = True
            reason = ""
        else:
            nForged = 0
            dynarray = create_empty_dynarray()
            successful = False
            reason = "No data available from any source"
        
        print(f"nForged/nTotal/expected {nForged}/{dynarray.shape[0]}/{nRowsExpected}. {reason}")
        
        if dynarray.shape[1] >= 7:
            dynarray = np.delete(dynarray, [0, 6], axis=1)
        dynarray = np.float32(dynarray)
        
        fetchedFromGstream = False
        successful = (dynarray.shape[0] == nRowsExpected)
        if not successful:
            reason = reason + ". The number of rows mismatches nRowsExpected."
        
        return dynarray, nForged, successful, fetchedFromGstream, reason
    
    def get_combined_history(self, datatype, interval, start, end, symbol_dict):
        features_to_return, nForged = [], 0
        
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
        
        return features_to_return, nForged
    
    def get_combined_history_2(self, datatype, interval, start, end, symbol_dict, target):
        features_to_return, nForged, target_feature_id = [], 0, None
        target_symbol, target_feature = target
        
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
        
        return features_to_return, nForged, target_feature_id
    
    def Stop_Maintain_GStreams(self):
        signaled = None
        if Binance.stop_Maintain_GStreams is not None:
            Binance.stop_Maintain_GStreams = True
            signaled = True
        else:
            signaled = False
        return signaled
    
    def Maintain_GStreams(self):
        Binance._running = True
        self.GStream_Transceiver_Thread_Launcher(restore=False)
        
        if Config['Run_Demisser']:
            self.GStream_Demisser_Thread_Launcher(restore=False)
        
        Binance.stop_Maintain_GStreams = False
        
        def stop_transceiver_thread():
            Binance.stop_GStream_Transceiver_Thread_Function = True
            Binance.gstream_transceiver_thread.join()
            Print('gstream_transceiver_thread joined.')
        
        def stop_demisser_thread():
            Binance.stop_GStream_Demisser_Thread_Function = True
            Binance.gstream_demisser_thread.join()
            Print('gstream_demisser_thread joined.')
        
        try:
            while True:
                tm.sleep(5)
                
                if Binance.stop_Maintain_GStreams is None:
                    pass
                elif Binance.stop_Maintain_GStreams:
                    Binance._running = False
                    
                    if Config['Run_Demisser'] and Binance.stop_GStream_Demisser_Thread_Function is not None:
                        if Binance.gstream_demisser_thread.is_alive():
                            stop_demisser_thread()
                    
                    if Binance.stop_GStream_Transceiver_Thread_Function is not None:
                        if Binance.gstream_transceiver_thread.is_alive():
                            stop_transceiver_thread()
                    break
                
                elif not Binance.stop_Maintain_GStreams:
                    if Config['Run_Demisser'] and Binance.demisser_gstream_heartbeat is not None \
                        and datetime.timestamp(datetime.now()) - Binance.demisser_gstream_heartbeat > 60:
                        Binance._running = False
                        if Binance.gstream_demisser_thread.is_alive():
                            stop_demisser_thread()
                        stop_GStream_Demisser_Thread_Function = False
                        self.GStream_Demisser_Thread_Launcher(restore=True)
                        Binance._running = True
                    
                    if Binance.transceiver_frame_heartbeat is not None \
                        and datetime.timestamp(datetime.now()) - Binance.transceiver_frame_heartbeat > 60:
                        Binance._running = False
                        if Binance.gstream_transceiver_thread.is_alive():
                            stop_transceiver_thread()
                        stop_GStream_Transceiver_Thread_Function = False
                        self.GStream_Transceiver_Thread_Launcher(restore=True)
                        Binance._running = True
        
        except Exception as e:
            Print('Maintain_GStreams() with exception: ')
            Binance.stop_Maintain_GStreams = None
            sys.exit(1)
        
        Print('Maintain_GStreams() stopped.')
        Binance.stop_Maintain_GStreams = None
        return
    
    def GStream_Transceiver_Thread_Launcher(self, restore=False):
        from unicorn_binance_websocket_api.manager import BinanceWebSocketApiManager as UnicornSockets
        Binance.unicorn_sockets_manager = UnicornSockets(exchange=Config['exchange'])
        Binance.gstream_transceiver_thread = threading.Thread(target=self.GStream_Transceiver_Thread_Function, args=())
        Binance.gstream_transceiver_thread.start()
        self.Dynamize_Static_GStreams_v2(live=True, restore=restore)
        self.Dynamize_Static_GStreams_v2(live=False, restore=restore)
        return
    
    def GStream_Transceiver_Thread_Function(self):
        sleepSec = 0.001
        Binance.stop_GStream_Transceiver_Thread_Function = False
        
        try:
            while Config['Stream_Price']:
                Binance.transceiver_frame_heartbeat = datetime.timestamp(datetime.now())
                nSleepSec = 0
                
                if Binance.stop_GStream_Transceiver_Thread_Function is None:
                    pass
                elif Binance.stop_GStream_Transceiver_Thread_Function:
                    Binance.unicorn_sockets_manager.stop_manager_with_all_streams()
                    break
                elif Binance.unicorn_sockets_manager.is_manager_stopping():
                    break
                elif not Binance.stop_GStream_Transceiver_Thread_Function:
                    frame = Binance.unicorn_sockets_manager.pop_stream_data_from_stream_buffer()
                    if frame is False:
                        time.sleep(sleepSec)
                        nSleepSec += sleepSec
                    else:
                        if type(frame) == str:
                            frame = json.loads(frame)
                        assert type(frame) == dict
                        
                        data_consumed = False
                        stream_str = frame.get('stream', None)
                        if stream_str is not None:
                            self.streaming = True
                            dataType, symbol, interval = Binance.Unicorn_dataType_Symbol_Interval(stream_str)
                            key = dataType + '.' + symbol + '.' + interval
                            
                            if Binance.gstreams.get(key, None) is None:
                                self.Maintain_Existence_Of_GStream(dataType, symbol, interval, live=True, max=200)
                            else:
                                data_consumed = True
                                frame = frame['data']
                                event_mili = frame['E']
                                candle_open_mili = frame['k']['t']
                                
                                Binance.gstream_datetime_utc = datetime.fromtimestamp(candle_open_mili / 1000).replace(tzinfo=pytz.utc)
                                utcnow = datetime.utcnow()
                                Binance.gstream_lag_sec = (utcnow - datetime.utcfromtimestamp(event_mili / 1000)).total_seconds()
                                
                                gs = Binance.gstreams[key]
                                live = bool(gs.t_state & T_State.live)
                                prev_closed = bool(gs.t_state & T_State.closed)
                                
                                if dataType == 'klines':
                                    assert symbol.upper() == frame['s']
                                    kstream = frame['k']
                                    now_closed = bool(kstream['x'])
                                    
                                    new_row = [kstream['t'], kstream['o'], kstream['h'], kstream['l'], kstream['c'], kstream['v'],
                                               kstream['T'], kstream['q'], kstream['n'], kstream['V'], kstream['Q'], kstream['B']]
                                    [new_row] = parse_datalines_to_datalines(dataType, [new_row])
                                    
                                    intervalmili = intervalToMilliseconds(interval)
                                    nExisting_rows = gs.dynarray.shape[0]
                                    overriden = None
                                    misses = None
                                    
                                    if live:
                                        if prev_closed:
                                            if nExisting_rows > 0:
                                                overriden = False
                                                misses = round((new_row[0] - gs.dynarray[-1, 0]) / intervalmili) - 1
                                                if misses >= 0:
                                                    gs.dynarray = np.append(gs.dynarray, [new_row], axis=0)
                                                    gs.t_state = gs.t_state | T_State.changed
                                            else:
                                                gs.dynarray = np.append(gs.dynarray, [new_row], axis=0)
                                                gs.t_state = gs.t_state | T_State.changed
                                        else:
                                            if nExisting_rows > 0:
                                                overriden = True
                                                misses = round((new_row[0] - gs.dynarray[-1, 0]) / intervalmili)
                                                if misses >= 0:
                                                    gs.dynarray[-1] = new_row
                                                    gs.t_state = gs.t_state | T_State.changed
                                            else:
                                                gs.dynarray = np.append(gs.dynarray, [new_row], axis=0)
                                                gs.t_state = gs.t_state | T_State.changed
                                    else:
                                        if prev_closed:
                                            if now_closed:
                                                if nExisting_rows > 0:
                                                    if round((new_row[0] - gs.dynarray[-1, 0]) / intervalmili) >= 1:
                                                        gs.dynarray = np.append(gs.dynarray, [new_row], axis=0)
                                                        gs.t_state = gs.t_state | T_State.changed
                                                else:
                                                    gs.dynarray = np.append(gs.dynarray, [new_row], axis=0)
                                                    gs.t_state = gs.t_state | T_State.changed
                                            else:
                                                pass
                                        else:
                                            if now_closed:
                                                if nExisting_rows > 0:
                                                    gs.dynarray[-1] = new_row
                                                else:
                                                    gs.dynarray = np.append(gs.dynarray, [new_row], axis=0)
                                                gs.t_state = gs.t_state | T_State.changed
                                            else:
                                                pass
                                    
                                    gs.t_state = (gs.t_state & ~T_State.closed) | T_State.closed if now_closed else gs.t_state & ~T_State.closed
                                    
                                    now_closed = bool(gs.t_state & T_State.closed)
                                    
                                    if Config['Show_Transceiver']:
                                        Print(
                                            frame['s'].ljust(10),
                                            frame['k']['i'].zfill(3),
                                            ', C' if now_closed else ',  ',
                                            ', N ' + utcnow.strftime('%M:%S'),
                                            ', R {:.0f}m'.format((utcnow.replace(tzinfo=pytz.utc) - Config['start_time_utc']).total_seconds() / 60),
                                            ', L {}s'.format(str(round(Binance.gstream_lag_sec))),
                                            ', S ' + datetime.utcfromtimestamp(candle_open_mili / 1000).strftime('%M:%S'),
                                            ', E ' + datetime.utcfromtimestamp(event_mili / 1000).strftime('%M:%S'),
                                            ', {}'.format('Over' if overriden else 'Next'),
                                            ', M {}'.format(misses),
                                        )
                                
                                if Binance.add_to_plot is not None:
                                    Binance.add_to_plot(symbol, candle_open_mili, new_row[4])
                        
                        else:
                            self.streaming = False
                        
                        time.sleep(0.01)
                        if not data_consumed:
                            pass
                
                if nSleepSec <= 0:
                    tm.sleep(sleepSec)
            
            return
        
        except Exception as e:
            Binance.gstream_transceiver_thread._is_stopped = True
            Binance.unicorn_sockets_manager.stop_manager_with_all_streams()
            sys.exit(1)
    
    def GStream_Demisser_Thread_Launcher(self, restore=False):
        Binance.gstream_demisser_thread = threading.Thread(target=self.GStream_Demisser_Thread_Function, args=())
        Binance.gstream_demisser_thread.start()
    
    def DemissGStream(self, key, nKeys, nMissings):
        successful = True
        nKeys1 = nKeys
        nMissings1 = nMissings
        next_val = None
        
        gs = Binance.gstreams[key]
        
        if key in Binance.badkeys:
            return nKeys, nMissings, "continue"
        nKeys += 1
        
        (dataType, symbol, interval) = key.split('.')
        monitor = symbol.ljust(12)
        Binance.demisser_gstream_heartbeat = datetime.timestamp(datetime.now())
        
        slot = get_timeslot(dataType)
        interval_mili = intervalToMilliseconds(interval)
        
        binance_wait_sec = 0 if nKeys < 180 else 1 if nKeys < 200 else Config['binance_wait_sec']
        
        no_missing = True
        live = bool(gs.t_state & T_State.live)
        closed = bool(gs.t_state & T_State.closed)
        
        initially_existing_rows = gs.dynarray.shape[0]
        search_start = 0
        search_end = initially_existing_rows - 1 if live and not closed else initially_existing_rows
        
        yesterday_first_ts_mili = round(datetime.timestamp(get_previoius_day_start(datetime.utcnow().replace(tzinfo=pytz.utc))) * 1000)
        
        if initially_existing_rows > 0:
            first_ts_mili_excl = round(gs.dynarray[0][slot])
        else:
            first_ts_mili_excl = round(datetime.timestamp(datetime.utcnow()) * 1000 / interval_mili) * interval_mili + interval_mili
            prepended_to_now = True
        
        nRowsToPrepend = 0
        if yesterday_first_ts_mili + interval_mili < first_ts_mili_excl:
            nRowsToPrepend = round((first_ts_mili_excl - yesterday_first_ts_mili) / interval_mili)
            if nRowsToPrepend > 0:
                rows = exact_fetch(self.clientPool, binance_wait_sec, dataType, symbol, interval, yesterday_first_ts_mili, first_ts_mili_excl - interval_mili, nRowsToPrepend, interval_mili)
                if isinstance(rows, list):
                    nRows = len(rows)
                    if nRows != nRowsToPrepend:
                        if Config['Show_Demisser']:
                            Print("??? Fetching failed. {} blacklisted.".format(key))
                        Binance.badkeys.append(key)
                        return nKeys, nMissings, "continue"
                    
                    try:
                        rows = parse_datalines_to_datalines(dataType, rows)
                        array = np.array(rows)
                        array = snap_sort_deduplicate_in_opening_time(array, interval_mili)
                        assert is_one_incremental(array[:, slot].astype(np.int64) / interval_mili), \
                            "The rows fetched for prepending is not one-incremental. {}".format(key)
                        
                        gs.dynarray = np.insert(gs.dynarray, 0, array, axis=0)
                        monitor += ", pre {:>4}".format(nRowsToPrepend)
                        search_start = nRowsToPrepend - 1
                    
                    except Exception as e:
                        Print("~~~~~~~~~~~~~~~~~~~~~~~ Exception in DemissGStream core: \n", str(e))
                else:
                    no_missing = False
                    monitor += ",".ljust(10)
            else:
                Print("~~~~~~~~~~~~~~~~~~~~~~~ Exception in DemissGStream: \n")
        
        elif first_ts_mili_excl < yesterday_first_ts_mili:
            gs.dynarray = gs.dynarray[gs.dynarray[:, slot] >= yesterday_first_ts_mili - 1]
            monitor += ",".ljust(10)
        
        if no_missing:
            nRows = initially_existing_rows + nRowsToPrepend
            search_end = nRows - 1 if live and not closed else nRows
            assert search_end >= 1, "search_end < 1."
            
            if (round(gs.dynarray[search_end - 1, slot]) - round(gs.dynarray[search_start, slot])) == (search_end - 1 - search_start) * interval_mili:
                assert is_one_incremental(gs.dynarray[search_start:search_end, slot].astype(np.int64) / interval_mili), \
                    "The rows following prepending is not one-incremental."
            else:
                gs.dynarray, nForged, no_missing = create_missing_prices_inland(
                    Binance.clientPool, binance_wait_sec, gs.dynarray, dataType, symbol, interval, 0, 0, ignore_last=live and closed
                )
                if nForged > 0:
                    monitor += ", F {}".format(nForged)
        
        if no_missing:
            try:
                assert is_one_incremental(gs.dynarray[0:search_end, slot].astype(np.int64) / interval_mili), \
                    "{} candles are not one-incremental. Correcting...".format(symbol)
            except:
                pass
        else:
            gs.t_state = gs.t_state | T_State.changed
        
        gs = Binance.gstreams[key]
        if no_missing != bool(gs.t_state & T_State.no_missing):
            if no_missing:
                gs.t_state = gs.t_state | T_State.no_missing
                monitor += ", No-missing".ljust(12)
            else:
                gs.t_state = gs.t_state & ~T_State.no_missing
                monitor += ", Missing".ljust(12)
                nMissings += 1
        elif bool(gs.t_state & T_State.no_missing):
            monitor += ", No-missing".ljust(12)
        else:
            monitor += ", Missing".ljust(12)
        
        monitor += ", nKey {:>3}, N {:>4}".format(nKeys, gs.dynarray.shape[0])
        monitor = "--- Demiss: " + monitor
        
        if Config['Show_Demisser']:
            Print(monitor)
        
        if Binance.stop_GStream_Demisser_Thread_Function:
            return nKeys, nMissings, "break"
        
        items = Binance.gstreams.items()
        nGStreams = len(items) if items is not None else 0
        tm.sleep(
            Config['Bias_Sec_per_Demiss'] / nGStreams
            + Config['Linear_Sec_per_Lag_X_#gstream'] * Binance.gstream_lag_sec * nGStreams
        )
        
        if Binance.gstream_datetime_utc is None:
            Binance.gstream_datetime_utc = from_dt_local_to_dt_utc(datetime.now())
        
        return nKeys, nMissings, "next"
    
    def GStream_Demisser_Thread_Function(self):
        Binance.stop_GStream_Demisser_Thread_Function = False
        
        try:
            while Config['Run_Demisser']:
                if Config['Show_Demisser']:
                    Print('******* new demisser cycle --------\n')
                
                nKeys = 0
                nMissings = 0
                
                if Binance.stop_GStream_Demisser_Thread_Function is None:
                    pass
                elif Binance.stop_GStream_Demisser_Thread_Function:
                    break
                elif not Binance.stop_GStream_Demisser_Thread_Function:
                    gstreams = Binance.gstreams.copy()
                    
                    for key, gs in gstreams.items():
                        nKeys, nMissings, next_val = self.DemissGStream(key, nKeys, nMissings)
                        
                        if next_val == "break":
                            break
                        if next_val == "continue":
                            continue
                        if next_val == "next":
                            pass
                
                if not Binance.stop_GStream_Demisser_Thread_Function:
                    if Config['Show_Demisser']:
                        Print('############# Demisser cycle done: missing {} of {} keys, with {} bad key(s)\n'.format(nMissings, nKeys, len(Binance.badkeys)))
                    if nMissings <= 0:
                        if Config['Show_Demisser']:
                            Print('------------GStreams Ready --------\n')
                        Binance.gstreams_ready = True
                    else:
                        Binance.gstreams_ready = False
        
        except Exception as e:
            Binance.gstream_demisser_thread._is_stopped = True
            Print("~~~~~~~~~~~~~~~~~~~~~~~ Exception in GStream_Demisser_Thread_Function: \n", str(e))
        
        return
    
    def Dynamize_Static_GStreams_v2(self, live=True, restore=False):
        Binance.gstreams = {key: value for (key, value) in sorted(Binance.gstreams.items(), key=lambda x: x[1].t_state & T_State.live, reverse=True)}
        
        slot_groups = []
        for key, gs in Binance.gstreams.items():
            if (bool(gs.t_state & T_State.live) == live) and (restore or not gs.t_state & (T_State.synced | T_State.syncing)):
                [dataType, symbol, interval] = key.split('.')
                market, channel = Binance.Unicorn_Market_Channle(dataType, symbol, interval)
                
                group_shared = False
                for slot_group in slot_groups:
                    channels = slot_group['channels']
                    markets = slot_group['markets']
                    if len(channels) * len(markets) < Config['max_subscriptions_per_stream']:
                        if len(channels) == 1 and channels[0] == channel:
                            markets.append(market)
                            group_shared = True
                        elif len(markets) == 1 and markets[0] == market:
                            channels.append(channel)
                            group_shared = True
                    if group_shared:
                        break
                
                if not group_shared:
                    slot_group = {'channels': [channel], 'markets': [market]}
                    slot_groups.append(slot_group)
        
        for slot_group in slot_groups:
            tm.sleep(Config['binance_wait_sec'])
            stream_id = self.unicorn_sockets_manager.create_stream(slot_group['channels'], slot_group['markets'])
            if Config['Show_Demisser']:
                Print("---{} gstreams on a new price stream".format(len(slot_group['channels']) * len(slot_group['markets'])))
            if stream_id is not None:
                for channel in slot_group['channels']:
                    for market in slot_group['markets']:
                        dataType, symbol, interval = Binance.Unicorn_dataType_Symbol_Interval_From_Channel_Market(channel, market)
                        key = dataType + '.' + symbol + '.' + interval
                        if Binance.gstreams.get(key, None) is not None:
                            Binance.gstreams[key].stream_id = stream_id
            else:
                raise Exception('Failed in an attempt to create stream.')
        return
    
    @classmethod
    def Start_Add_To_Plot(cls, add_to_plot):
        cls.add_to_plot = add_to_plot
    
    def GetExchangeTimeMili(cls):
        server_time = cls.client.get_server_time()
        miliseconds_from_epoch = round(server_time["serverTime"])
        return miliseconds_from_epoch
    
    def Maintain_Existence_Of_GStream(self, dataType, symbol, interval, live=False, max=200):
        symbol = str.upper(symbol)
        interval = str.lower(interval)
        key = dataType + '.' + symbol + '.' + interval
        
        gs = None
        if Binance.gstreams.get(key, None) is not None:
            gs = Binance.gstreams[key]
            gs.t_state = gs.t_state | T_State.live if live else gs.t_state & ~T_State.none
        elif len(Binance.gstreams.items()) < max:
            gs = self.Create_Static_GStream_v2(dataType, symbol, interval, live=live)
            Binance.gstreams[key] = gs
            if Config['Show_Transceiver'] or Config['Show_Traders']:
                Print("+++++ {} is newly added to gstreams.".format(key))
            
            if Binance.unicorn_sockets_manager is not None:
                if Binance._running and gs is not None:
                    if Config['Show_Transceiver'] or Config['Show_Traders']:
                        Print("+++++ {} is being added to Binance streams...".format(key))
                    self.Subscribe_to_Existing_or_New_Active_Stream(key)
        
        return key
    
    def Create_Static_GStream_v2(self, dataType, symbol, interval, live=False):
        dynarray = create_empty_dynarray()
        filepath, stream_id = None, None
        tickbuffer, t_state = [], T_State.none | (T_State.live if live else T_State.none) | T_State.closed | T_State.no_missing
        return gstream(threading.Lock(), dynarray, filepath, stream_id, tickbuffer, t_state)
    
    def Unsubscribe_GStream(self, dataType, symbol, interval):
        symbol = str.upper(symbol)
        interval = str.lower(interval)
        unsubscribed = False
        
        market, channel = Binance.Unicorn_Market_Channle(dataType, symbol, interval)
        
        with Binance.stream_lock:
            stream_ids = [id for (id, _) in Binance.unicorn_sockets_manager.get_stream_list().items()]
            for stream_id in stream_ids:
                stream_info = Binance.unicorn_sockets_manager.get_stream_info(stream_id)
                if stream_info['channels'] is not None and len(stream_info['channels']) == 1 and stream_info['channels'][0] == channel:
                    unsubscribed = Binance.unicorn_sockets_manager.unsubscribe_from_stream(stream_id, [], [market])
                elif stream_info['markets'] is not None and len(stream_info['market']) == 1 and stream_info['markets'][0] == market:
                    unsubscribed = Binance.unicorn_sockets_manager.unsubscribe_from_stream(stream_id, [channel], [])
        return
    
    def Is_On_Active_Streams(self, dataType, symbol, interval):
        symbol = str.upper(symbol)
        interval = str.lower(interval)
        
        market, channel = Binance.Unicorn_Market_Channle(dataType, symbol, interval)
        
        found_active = False
        stream_list = Binance.unicorn_sockets_manager.get_stream_list()
        if stream_list is not None and isinstance(stream_list, dict):
            with Binance.stream_lock:
                stream_ids = [id for (id, _) in Binance.unicorn_sockets_manager.get_stream_list().items()]
                for stream_id in stream_ids:
                    stream_info = Binance.unicorn_sockets_manager.get_stream_info(stream_id)
                    if stream_info['channels'] is not None and stream_info['markets'] is not None:
                        if channel in stream_info['channels'] and market in stream_info['markets']:
                            found_active = True
                            break
        
        return found_active
    
    def Remove_Duplicates_On_Active_Streams(self, dataType, symbol, interval):
        symbol = str.upper(symbol)
        interval = str.lower(interval)
        unsubscribed = False
        
        market, channel = Binance.Unicorn_Market_Channle(dataType, symbol, interval)
        
        found_active = False
        with Binance.stream_lock:
            stream_ids = [id for (id, _) in Binance.unicorn_sockets_manager.get_stream_list().items()]
            for stream_id in stream_ids:
                stream_info = Binance.unicorn_sockets_manager.get_stream_info(stream_id)
                if channel in stream_info['channels'] and market in stream_info['markets']:
                    if not found_active:
                        found_active = True
                    else:
                        if stream_info['channels'] is not None and len(stream_info['channels']) == 1 and stream_info['channels'][0] == channel:
                            unsubscribed = Binance.unicorn_sockets_manager.unsubscribe_from_stream(stream_id, [], [market])
                        elif stream_info['markets'] is not None and len(stream_info['market']) == 1 and stream_info['markets'][0] == market:
                            unsubscribed = Binance.unicorn_sockets_manager.unsubscribe_from_stream(stream_id, [channel], [])
        
        return
    
    def Remove_GStream(self, dataType, symbol, interval):
        Print("-----------Remove_GStream")
        symbol = str.upper(symbol)
        interval = str.lower(interval)
        
        self.Unsubscribe_GStream(dataType, symbol, interval)
        
        key = dataType + '.' + symbol + '.' + interval
        if Binance.gstreams.get(key, None) is not None:
            del Binance.gstreams[key]
        
        return
    
    def Download_Monthly_Data(self, dataTypes, symbols, intervals, start, end):
        start_utc = from_dt_local_to_dt_utc(start)
        if end is None:
            end = datetime.now()
        end_utc = from_dt_local_to_dt_utc(end)
        
        start_utc = datetime(start_utc.year, start_utc.month, 1) - timedelta(days=1)
        end_utc = datetime(end_utc.year, end_utc.month + 1, 1)
        
        paths = {}
        symbolcounter = 0
        for dataType in dataTypes:
            for symbol in symbols:
                Print("[{}/{}] - start download monthly {} {}.".format(symbolcounter + 1, len(symbols), symbol, dataType))
                for interval in intervals:
                    paths_list = []
                    for year in range(start_utc.year, end_utc.year + 1):
                        for month in range(1, 13):
                            current_utc = datetime(year, month, 2)
                            if start_utc <= current_utc and current_utc <= end_utc:
                                path = download_file(current_utc, dataType, symbol, interval)
                                paths_list.append(path)
                    paths[dataType + '.' + symbol + '.' + interval] = paths_list
                symbolcounter += 1
        
        return paths
    
    def Download_Daily_Data(self, dataTypes, symbols, intervals, start, end):
        start_utc = from_dt_local_to_dt_utc(start)
        if end is None:
            end = datetime.now()
        end_utc = from_dt_local_to_dt_utc(end)
        
        start_utc = datetime(start_utc.year, start_utc.month, start_utc.day) - timedelta(hours=1)
        end_utc = datetime(end_utc.year, end_utc.month, end_utc.day) + timedelta(hours=1)
        
        paths = {}
        symbolcounter = 0
        for dataType in dataTypes:
            for symbol in symbols:
                Print("[{}/{}] - start download daily {} {}.".format(symbolcounter + 1, len(symbols), symbol, dataType))
                for interval in intervals:
                    paths_list = []
                    for days in range((end_utc - start_utc).days + 2):
                        current_utc = datetime(start_utc.year, start_utc.month, start_utc.day) + timedelta(days=days)
                        if start_utc <= current_utc and current_utc <= end_utc:
                            path = download_file(current_utc, dataType, symbol, interval)
                            paths_list.append(path)
                    paths[dataType + '.' + symbol + '.' + interval] = paths_list
                symbolcounter += 1
        
        return paths
    
    @staticmethod
    def Unicorn_Market_Channle(dataType, symbol, interval):
        market = str.lower(symbol)
        if dataType == 'klines':
            channel = 'kline_' + interval
        elif dataType == 'aggTrades':
            channel = 'aggTrade'
        else:
            raise Exception('Unhandled data type: {}'.format(dataType))
        return market, channel
    
    @staticmethod
    def Unicorn_dataType_Symbol_Interval_From_Channel_Market(channel, market):
        symbol = str.upper(market)
        if '_' in channel:
            [dataType, interval] = channel.split('_')
            if dataType == 'kline':
                dataType = 'klines'
        elif channel == 'aggTrades':
            dataType = 'aggTrade'
            interval = None
        else:
            raise Exception('Unhandled channel: {}'.format(channel))
        return dataType, symbol, interval
    
    @staticmethod
    def Unicorn_dataType_Symbol_Interval(stream_str):
        dataType, symbol, interval = None, None, None
        [market, channel] = stream_str.split('@')
        symbol = str.upper(market)
        if '_' in channel:
            [dataType, interval] = channel.split('_')
        else:
            dataType = channel
        if dataType == 'kline':
            dataType = 'klines'
        return dataType, symbol, interval
    
    def Subscribe_to_Existing_or_New_Active_Stream_v2(self, key):
        [dataType, symbol, interval] = key.split('.')
        market, channel = Binance.Unicorn_Market_Channle(dataType, symbol, interval)
        stream_id = Binance.unicorn_sockets_manager.create_stream(channel, market)
        return stream_id
    
    def Subscribe_to_Existing_or_New_Active_Stream(self, key):
        [dataType, symbol, interval] = key.split('.')
        market, channel = Binance.Unicorn_Market_Channle(dataType, symbol, interval)
        
        stream_shared = False
        with Binance.stream_lock:
            stream_ids = None
            try:
                stream_list = copy.deepcopy(Binance.unicorn_sockets_manager.get_stream_list())
                stream_ids = [id for (id, _) in stream_list.items()]
            except:
                pass
            
            if stream_ids is not None and isinstance(stream_ids, list) and len(stream_ids) > 0:
                subscription = None
                for stream_id in stream_ids:
                    stream_info = Binance.unicorn_sockets_manager.get_stream_info(stream_id)
                    if stream_info['channels'] is not None and stream_info['markets'] is not None:
                        if len(stream_info['channels']) * len(stream_info['markets']) < Config['max_slots_per_stream']:
                            if len(stream_info['channels']) == 1 and stream_info['channels'][0] == channel:
                                subscription = (stream_id, [], [market])
                                stream_shared = True
                            elif len(stream_info['market']) == 1 and stream_info['markets'][0] == market:
                                subscription = (stream_id, [channel], [])
                                stream_shared = True
                            
                            if stream_shared:
                                break
                
                stream_id = None
                if subscription is not None:
                    Binance.unicorn_sockets_manager.subscribe_to_stream(subscription[0], subscription[1], subscription[2])
                    stream_id = subscription[0]
                else:
                    stream_id = Binance.unicorn_sockets_manager.create_stream(channel, market)
            else:
                stream_id = Binance.unicorn_sockets_manager.create_stream(channel, market)
        
        return stream_id
    
    def Get_Price_Data_By_Time_Batch(self, dataType, symbols, interval, start, end, lookup=True):
        table = []
        nForged = 0
        successful = True
        nPoints = None
        reports = []
        nSuccess = 0
        
        Print("Walking through the list...")
        
        counter = 1
        for symbol in symbols:
            dyarray, _nForged, successful, _, reason = \
                self.Get_Price_Data_By_Time(dataType, symbol, interval, start, end, lookup=lookup)
            
            report = symbol + ": "
            if successful:
                if nPoints is None:
                    nPoints = dyarray.shape[0]
                elif nPoints != dyarray.shape[0]:
                    successful = False
                    nForged += _nForged
                
                report += str(dyarray.shape[0]) + ", {} created".format(_nForged)
                
                if nPoints == dyarray.shape[0]:
                    table.append(dyarray)
                    nForged += _nForged
                    report += ". Success. {}-th.".format(counter)
                    nSuccess += 1
                else:
                    report += ". Failure. {}-th. ".format(counter) + reason
                counter += 1
                
                Print("\n", report)
            
            reports.append(report)
        
        if len(table) > 0:
            table = np.stack(table, axis=0)
        else:
            table = np.array([])
        
        Print("\nSuccessful for {}/{} markets".format(nSuccess, len(symbols)))
        
        successful = len(symbols) == nSuccess
        return table, nForged, reports, successful
    
    def Get_Cached_Price(self, dataType, symbol, interval, start):
        array = None
        key = dataType + '.' + symbol + '.' + interval
        if interval == "1m" and Binance.gstreams.get(key, None) is not None and key not in Binance.badkeys:
            gs = Binance.gstreams[key]
            no_missing = bool(gs.t_state & T_State.no_missing)
            if no_missing:
                from_utc = from_dt_local_to_dt_utc(start)
                array = select_from_dynarray_by_time(dataType, gs.dynarray, from_utc, None)
        return array
    
    def UpdateLookupState(self, firstOpenTimeMile, _latestDataStart, _checkedDataStart, start_lookedup):
        if _latestDataStart is None:
            _latestDataStart = datetime.fromtimestamp(firstOpenTimeMile / 1000).replace(tzinfo=pytz.utc)
        if not start_lookedup:
            _checkedDataStart = datetime.fromtimestamp(firstOpenTimeMile / 1000).replace(tzinfo=pytz.utc)
            start_lookedup = True
        return _latestDataStart, _checkedDataStart, start_lookedup
    
    def Place_Order(self):
        return

# ==================== Main Entry Point ====================

if __name__ == "__main__":
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