import math
import random
import numpy as np

def abs_normalize(arr):
    arr = arr - arr.min()
    arr = arr / (arr.std() + 1e-9)
    return arr

epsilon = 1e-12

def get_back_mean_std(arr, anchor, back, axis=0):   # [anchor - back + 1, anchor], n = back
    v = arr[max(0, anchor - back + 1): anchor + 1]      
    return v.mean(axis=axis), v.std(axis=axis)

def get_zscore(arr, anchor, back, axis=0):  # [anchor - back + 1, anchor], n = back
    sma, std = get_back_mean_std(arr, anchor, back, axis=axis)
    z_score = (arr[anchor] - sma) / (std + epsilon)
    return z_score

def get_level(arr, anchor, back, axis=0):   # [anchor - (back - 1), anchor], n = back
    return get_zscore(arr, anchor, back, axis=axis)

def get_change(arr, anchor, back, axis=0):    # [anchor - 2 * (back - 1), anchor], n = 2 * back - 1, safe init anchor = 2 * (back - 1)
    pcs = []
    for idx in range(max(0, anchor - back + 1), anchor + 1):
       sma = arr[max(0, idx - back + 1) : idx + 1].mean()   # [anchor - 2 * back + 2, anchor]
       pc = (arr[idx] / sma) - 1
       pcs.append(pc)
    pcs = np.array(pcs)
    assert len(pcs) <= back     # = is required
    price_change = get_zscore(pcs, len(pcs) - 1, back, axis=axis)
    return price_change

def normalize(arr, axis=0):
    return (arr - arr.mean(axis=axis)) / (arr.std(axis=axis) + epsilon)
    
def ema(arr, span): # [0, len(arr) - 1 ], n = len(arr)
    """
    Simple function to get final EMA value
    """
    alpha = 2 / (span + 1)
    ema = arr[0]
    for value in arr[1:]:
        ema = alpha * value + (1 - alpha) * ema
    return ema
    
def get_volatility(arr, anchor, back, axis=0, span=20):     # [anchor - 3 * back + 2, anchor], n = 3 * back - 1, safe init anchor = 3 * back - 2
    abs_volatilities = []
    for idx in range(max(0, anchor - back + 1), anchor + 1):    # [anchor - back + 1, anchor], n = back
        span_curr = arr[max(0, idx - back + 1) : max(0, idx + 1)]       # [anchor - 2 * back + 2, anchor]
        ema_curr = ema(span_curr, span) if span_curr.shape[0] > 0 else arr[idx]
        span_past = arr[max(0, idx - 2 * back + 1) : max(0, idx - back + 1)]    # [anchor - 3 * back + 2, anchor - back]
        ema_past =  ema(span_past, span) if span_past.shape[0] > 0 else arr[idx]
        volatility = (ema_curr - ema_past) / (ema_past + epsilon)
        abs_volatilities.append(volatility)
    assert len(abs_volatilities) <= back    # = is required
    abs_volatilities = np.array(abs_volatilities)
    # print(f"abs_volatilities: {len(abs_volatilities)}")
    volatility = get_zscore(abs_volatilities, len(abs_volatilities) - 1, back, axis=axis)
    return volatility
