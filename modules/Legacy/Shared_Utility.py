import sys
from datetime import datetime
import time as tm
from config import *
import numpy as np
from binance.client import Client as BinanceClient

import requests

def Call_Binance_API_v2(clientPool, method_name, nTrials, wait_sec, *args, **kwargs):
    # https://python-binance.readthedocs.io/en/latest/binance.html#binance.client.Client.get_klines
    result = None

    successful = False

    while not successful and nTrials > 0:
        client = None
        try:
            now_utc = datetime.utcnow().replace(tzinfo = pytz.utc)
            ts_mili= round(datetime.timestamp(now_utc)*1000)
            client = clientPool.Next()
            method = getattr(BinanceClient, method_name)
            result = method(client, *args, **kwargs) # I suspect Binance sometimes holds this call and wouldn't return, as a penalty.
            successful = True
        except:
            print("Oops!", sys.exc_info()[0], "occurred.")  # no Print, as sys.exc_info()[0] is not a string.
            clientPool.Repair(client)
            tm.sleep(wait_sec)
            nTrials -= 1
            if nTrials > 0: continue

    return result, ts_mili


def exact_fetch(clientPool, binance_wait_sec, dataType, symbol, interval, startTime, endTime_inclusive, nToFetch, interval_mili):
  # startTime is the opentime of the first candle to fetch.
  # endTime_inclusive is the opentime of the last candle to fetch

  datalines = []
  lastTime = startTime - interval_mili

  while lastTime < endTime_inclusive:
    maxLimit = 500 # min(500, nToFetch)

    if dataType == 'klines':
        #  https://python-binance.readthedocs.io/en/latest/binance.html#binance.client.Client.get_klines
        #  startTime is inclusive.
        #  We assume endtime is inclusive. Because https://github.com/sammchardy/python-binance/blob/master/binance/client.py says 
        #  get_aggregate_trades's endTime_inclusive is inclusive.
        nTrials = 1
        lines, ts_mili = Call_Binance_API_v2(clientPool, "get_klines", nTrials, binance_wait_sec, symbol = symbol, interval = interval, limit = maxLimit, startTime = lastTime + interval_mili, endTime = endTime_inclusive)
        #lines = clientPool.Next().get_klines(symbol = symbol, interval = interval, limit = maxLimit, startTime = lastTime + interval_mili, endTime = endId)
    elif dataType == 'trafes':
      lines = None

    elif dataType == 'aggTrades':   # FIX it.
        lines = clientPool.Next().get_aggregate_trades(symbol = symbol, fromId = lastTime + 1, startTime = lastTime + interval_mili, endTime = endTime_inclusive, limit = maxLimit)

    if lines is not None and isinstance(lines, list) and len(lines) > 0:
        datalines += lines
        lastTime = lines[-1][0]
    else:
        datalines = None
        break

  # Note: Most of the time, the last line is under construction if dataType == 'klines' or 'aggTrades'?
  # Do NOT remove them, though, because it sometimes has just been completed.

  return datalines

def is_one_incremental(array): # array: 1-d numpy array
    yes = None
    if array.shape[0] < 2:
        yes = True
    else:
        delta = array[1: ] - array[:-1]
        # Very narrow constraint... but a huge contribution to the integrity.
        yes = round(delta.max(axis=0)) == 1 and round(delta.min(axis=0)) == 1 

    return yes

def starting_index_of_1st_gap(array):      # comeback - overriding the below
    """
    Find the last index where the array has consecutive values (delta == 1).
    Returns the index of the last element before the first gap, or None if no gap exists.
    array: 1-d numpy array of time values divided by interval_mili (should be integers)
    """
    if array.shape[0] < 2:
        return None
    
    delta = array[1:] - array[:-1]
    # Find the first gap where delta > 1
    gap_indices = np.where(delta > 1)[0]
    
    if len(gap_indices) == 0:
        # No gaps, all are consecutive
        return None
    
    first_gap_idx = gap_indices[0]
    # Return the index of the last element before the gap
    return int(first_gap_idx)

# def starting_index_of_1st_gap(array): # array: 1-d numpy array
#     index = None
#     if array.shape[0] < 2:
#         pass
#     else:
#         delta = array[1: ] - array[:-1]
#         index = int(np.argmax( delta > 1 ))     # int, because of numpy
#         if index == 0:
#             if delta[index] == 1:
#                 index = None

#     return index


coincodex_candle_sec = 300
history_format = "https://coincodex.com/api/coincodex/get_coin_history/{}/{}/{}/{}"

def get_coincodex_history_numpy(symbol, start, end):
    result = None
    assert start <= end
    start -= timedelta(seconds=coincodex_candle_sec)
    end += timedelta(seconds=coincodex_candle_sec)
    start_date = "-".join([str(start.year%100), str(start.month).zfill(2), str(start.day).zfill(2)])
    end_date = "-".join([str(end.year%100), str(end.month).zfill(2), str(end.day).zfill(2)])
    delta = end - start
    samples = round( (delta.days + 1) * 86400 / coincodex_candle_sec + 0.1)    # 86400 seconds per day, coincodex_candle_sec seconds per coincodex candle.
    history_url = history_format.format(symbol, start_date, end_date, samples)
    try:
        response = requests.put(history_url)
        response.raise_for_status()
        try:
            result = response.json()[symbol]
            result = np.array(result)
        except:
            pass
    except requests.exceptions.HTTPError as errh:
        print(errh)
    except requests.exceptions.ConnectionError as errc:
        print(errc)
    except requests.exceptions.Timeout as errt:
        print(errt)
    except requests.exceptions.RequestException as err:
        print(err)

    if result is not None:
        try:
            assert result.shape[0] > 0
            assert result[0, 0] % (coincodex_candle_sec) == 0
            if result.shape[0] > 1:    # Check if timestamps are correct.
                delta = result[1:, 0] - result[:-1, 0]
                assert np.min(delta) == np.max(delta)
        except:
            result = None
    return result


def extract_from_coincodex_history(history, start, end, interval_mili):

    start_sec = round(datetime.timestamp(start)*1000/interval_mili) * round(interval_mili/1000)
    end_sec = round(datetime.timestamp(end)*1000/interval_mili) * round(interval_mili/1000)

    adds = int(coincodex_candle_sec/(interval_mili/1000)) - 1
    if adds > 0:
        history = history[ np.logical_and( start_sec - adds * interval_mili/1000 <= history[:, 0], history[:, 0] <= end_sec + adds * interval_mili/1000 ) ]
        for t in range(history.shape[0]-1, 0, -1):
            h = history[t]
            if t > 0: h_ = history[t-1]
            hs = [h.copy() for _ in range(adds)] # Not [h for _ in range(adds)]   # Not [h] x add
            # print("ts", h[0],  interval_mili/1000)
            for i in range(adds):
                hs[i][0] = h[0] - (adds-i) * interval_mili/1000
                if t > 0:
                    hs[i][1] = ( h_[1] * (adds-i) + h[1] * (i+1) ) / (adds+1)   # price is interploated.
            hs = np.array(hs)
            # print(history.shape, hs.shape)
            history = np.insert(history, t, hs, axis=0)

    history = history[ np.logical_and( start_sec <= history[:,0], history[:,0] <= end_sec) ]
    try:
        assert history[0, 0] == start_sec
        assert end_sec == history[-1, 0]
        assert history[0, 0] % (interval_mili/1000) == 0
        if history.shape[0] > 1:
            delta = history[1:,0] - history[:-1,0]
            assert delta[0] == interval_mili/1000
            assert np.min(delta) == np.max(delta)
        assert history.shape[0] == round((end_sec - start_sec)/(interval_mili/1000)) + 1
    except:
        history = None

    return history


ccInterval_sec = 300  # !!!!!!!!!! 300 or 60, it doesn't affect the result !!!!!!!
ccHistoryFormat = "https://coincodex.com/api/coincodex/get_coin_history/{}/{}/{}/{}"

def fetch_coincodex_history_numpy(symbol, start, end, clientInterval_sec):  # !!!!!!!!!! Coincodex call is extremely unpredictable !!!!!!!!!!!!!
    result = None
    assert start <= end
    start_sec = round(datetime.timestamp(start)/clientInterval_sec) * clientInterval_sec
    end_sec = round(datetime.timestamp(end)/clientInterval_sec) * clientInterval_sec
    start_clientCandle = round(start_sec / clientInterval_sec)
    end_clientCandle = round(end_sec / clientInterval_sec)
    start_cc = datetime.fromtimestamp( (int(clientInterval_sec * start_clientCandle / ccInterval_sec ) + 0 ) * ccInterval_sec )
    end_cc = datetime.fromtimestamp( (int(clientInterval_sec  * end_clientCandle / ccInterval_sec) + 1 ) * ccInterval_sec)

    start_date = "-".join([str(start_cc.year%100), str(start_cc.month).zfill(2), str(start_cc.day).zfill(2)])
    end_date = "-".join([str(end_cc.year%100), str(end_cc.month).zfill(2), str(end_cc.day).zfill(2)])
    delta = end - start
    samples = int( (delta.days + 1) * 86400 / ccInterval_sec + 0.1)    # 86400 seconds per day, ccInterval_sec seconds per cc candle.
    history_url = ccHistoryFormat.format(symbol, start_date, end_date, samples)

    try:        
        # response: 0: Unix timestamp, 1: price in USD (no more specification), 2: 24-hour volume, 3: market cap in usd.
        response = requests.put(history_url)
        # response = requests.get(history_url)      # Getting insane, as this line wouldn't work.        
        response.raise_for_status()
        result = response.json()[symbol]
        result = np.array(result)
    except:
        result = None

    if result is not None:
        # print(f"Coincodex data: {result.shape}")
        try:
            assert result.shape[0] > 0
            assert round(result[0, 0]) % (ccInterval_sec) == 0
            if result.shape[0] > 1:    # Check if timestamps are correct.
                delta = result[1:, 0] - result[:-1, 0]
                assert np.min(delta) == np.max(delta)
        except:
            result = None
    
    history = None
    if result is not None:
        left_to_start = int(clientInterval_sec * start_clientCandle / ccInterval_sec)
        offset = np.argmax( np.round(result[:, 0]) == left_to_start * ccInterval_sec )
        try:
            assert round(result[offset, 0]) == left_to_start * ccInterval_sec
            history = np.zeros( (end_clientCandle - start_clientCandle + 1, 4), dtype=float )
            for i in range(start_clientCandle, end_clientCandle + 1, 1):
                left_to_i = int(clientInterval_sec * i / ccInterval_sec)
                id_left =  left_to_i - left_to_start + offset
                client_sec = i * clientInterval_sec
                cc_left_sec = left_to_i * ccInterval_sec
                cc_right_sec = cc_left_sec + ccInterval_sec
                try:
                    assert result[id_left, 0] <= client_sec
                    assert client_sec <= result[id_left + 1, 0]
                    assert round(result[id_left, 0]) == cc_left_sec
                    assert round(result[id_left+1, 0]) == cc_right_sec

                    history[i-start_clientCandle, 0] = client_sec
                    history[i-start_clientCandle, 1:4] = ( (cc_right_sec - client_sec) * result[id_left, 1:4] + (client_sec - cc_left_sec) * result[id_left+1, 1:4] ) / ccInterval_sec
                except:
                    history = None
                    break
            
            if history is not None:
                assert history[0, 0] == start_sec
                assert end_sec == history[-1, 0]
                assert history[0, 0] % clientInterval_sec == 0
                if history.shape[0] > 1:
                    delta = history[1:,0] - history[:-1,0]
                    assert delta[0] == clientInterval_sec
                    assert np.min(delta) == np.max(delta)
                assert history.shape[0] == end_clientCandle - start_clientCandle + 1
        except:
            history = None

    return history
