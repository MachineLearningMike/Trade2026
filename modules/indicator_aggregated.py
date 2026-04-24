import numpy as np
from typing import Dict

def calculate_best_indicators(price: np.ndarray, volume: np.ndarray) -> Dict[str, float]:
    """
    Calculate best trading indicators as scalar values.
    ALL indicators are normalized to range [-1, 1].
    
    Parameters:
    -----------
    price : np.ndarray
        Close price array (full history)
    volume : np.ndarray
        Volume array (full history)
    
    Returns:
    --------
    dict : Indicator name -> normalized scalar value in [-1, 1]
    """
    
    price = np.asarray(price).flatten()
    volume = np.asarray(volume).flatten()
    
    indicators = {}
    
    # ========== TREND INDICATORS ==========
    
    # 1. EMA 20 - Normalized [-1, 1]
    ema_20 = _ema(price, 20)
    indicators['ema_20'] = _normalize_ratio(price[-1] / ema_20[-1], 0.95, 1.05)
    
    # 2. EMA 50 - Normalized [-1, 1]
    ema_50 = _ema(price, 50) if len(price) >= 50 else ema_20
    indicators['ema_50'] = _normalize_ratio(price[-1] / ema_50[-1], 0.95, 1.05)
    
    # 3. EMA 200 - Normalized [-1, 1]
    if len(price) >= 200:
        ema_200 = _ema(price, 200)
        indicators['ema_200'] = _normalize_ratio(price[-1] / ema_200[-1], 0.95, 1.05)
    else:
        indicators['ema_200'] = 0.0
    
    # 4. MACD Histogram - Normalized [-1, 1]
    _, _, histogram = _macd(price, 12, 26, 9)
    hist_std = np.std(histogram[-100:]) if len(histogram) >= 100 else 1.0
    indicators['macd_histogram'] = np.clip(np.tanh(histogram[-1] / (hist_std + 1e-9)), -1, 1)
    
    # 5. ADX - Normalized [-1, 1] (0 = no trend, 1 = strong trend, but mapped to -1..1)
    adx = _adx_close(price, 14)
    indicators['adx'] = np.clip((adx[-1] - 25) / 25, -1, 1)  # 0->-1, 25->0, 50->1
    
    # 6. DI Difference - Normalized [-1, 1]
    _, plus_di, minus_di = _adx_full(price, 14)
    di_diff_raw = (plus_di[-1] - minus_di[-1]) / 50.0
    indicators['di_diff'] = np.clip(di_diff_raw, -1, 1)
    
    # ========== MOMENTUM INDICATORS ==========
    
    # 7. RSI - Normalized [-1, 1] (converted from 0-100)
    rsi = _rsi(price, 14)
    indicators['rsi'] = (rsi[-1] - 50) / 50
    
    # 8. Stochastic K - Normalized [-1, 1]
    stoch_k, _ = _stochastic_close(price, 14, 3)
    indicators['stoch_k'] = (stoch_k[-1] - 50) / 50
    
    # 9. Stochastic D - Normalized [-1, 1]
    _, stoch_d = _stochastic_close(price, 14, 3)
    indicators['stoch_d'] = (stoch_d[-1] - 50) / 50
    
    # 10. CCI - Normalized [-1, 1] (tanh of CCI/100)
    cci = _cci_close(price, 20)
    indicators['cci'] = np.tanh(cci[-1] / 100.0)
    
    # 11. ROC - Normalized [-1, 1]
    roc = _roc(price, 12)
    indicators['roc'] = np.tanh(roc[-1] / 10.0)
    
    # 12. Williams %R - Normalized [-1, 1] (converted from -100 to 0)
    williams_r = _williams_r_close(price, 14)
    indicators['williams_r'] = (williams_r[-1] + 50) / 50  # -100->-1, -50->0, 0->1
    
    # ========== VOLATILITY INDICATORS ==========
    
    # 13. Bollinger Band Position - Normalized [-1, 1]
    bb_upper, bb_middle, bb_lower = _bollinger_bands(price, 20, 2)
    bb_range = bb_upper[-1] - bb_lower[-1]
    if bb_range > 0:
        bb_pos = (price[-1] - bb_middle[-1]) / (bb_range / 2)
        indicators['bb_position'] = np.clip(bb_pos, -1, 1)
    else:
        indicators['bb_position'] = 0.0
    
    # 14. Bollinger Band Width - Normalized [-1, 1] (squeeze indicator)
    bb_width_raw = (bb_upper[-1] - bb_lower[-1]) / (bb_middle[-1] + 1e-9)
    indicators['bb_width'] = np.clip((bb_width_raw - 0.02) / 0.08, -1, 1)
    
    # 15. ATR Ratio - Normalized [-1, 1]
    atr = _atr_close(price, 14)
    atr_ratio_raw = atr[-1] / (price[-1] + 1e-9)
    indicators['atr_ratio'] = np.clip((atr_ratio_raw - 0.01) / 0.04, -1, 1)
    
    # 16. Keltner Channel Position - Normalized [-1, 1]
    kc_upper, kc_middle, kc_lower = _keltner_channel_close(price, 20, 1.5)
    kc_range = kc_upper[-1] - kc_lower[-1]
    if kc_range > 0:
        kc_pos = (price[-1] - kc_middle[-1]) / (kc_range / 2)
        indicators['kc_position'] = np.clip(kc_pos, -1, 1)
    else:
        indicators['kc_position'] = 0.0
    
    # ========== VOLUME INDICATORS ==========
    
    # 17. OBV Ratio - Normalized [-1, 1]
    obv = _obv(price, volume)
    obv_ema = _ema(obv, 20)
    obv_ratio_raw = obv[-1] / (abs(obv_ema[-1]) + 1e-9)
    indicators['obv_ratio'] = np.clip(np.tanh(obv_ratio_raw - 1), -1, 1)
    
    # 18. Volume Ratio - Normalized [-1, 1]
    vol_ratio_raw = volume[-1] / (np.mean(volume[-20:]) + 1e-9)
    indicators['volume_ratio'] = np.clip((vol_ratio_raw - 1) / 2, -1, 1)
    
    # 19. VWAP Distance - Normalized [-1, 1]
    vwap = _vwap(price, volume)
    vwap_dist_raw = (price[-1] - vwap[-1]) / (vwap[-1] + 1e-9)
    indicators['vwap_distance'] = np.clip(np.tanh(vwap_dist_raw * 10), -1, 1)
    
    # 20. MFI - Normalized [-1, 1] (converted from 0-100)
    mfi = _mfi(price, volume, 14)
    indicators['mfi'] = (mfi[-1] - 50) / 50
    
    # 21. Chaikin Money Flow - Normalized [-1, 1]
    cmf = _chaikin_money_flow_close(price, volume, 20)
    indicators['cmf'] = np.clip(cmf[-1], -1, 1)
    
    return indicators


# ========== NORMALIZATION HELPER ==========

def _normalize_ratio(value: float, low: float = 0.95, high: float = 1.05) -> float:
    """
    Normalize a ratio to [-1, 1] range with typical bounds.
    value < low -> -1, value > high -> 1, linear mapping in between.
    """
    if value <= low:
        return -1.0
    elif value >= high:
        return 1.0
    else:
        # Map [low, high] -> [-1, 1]
        return (value - low) / (high - low) * 2 - 1


# ========== INDICATOR IMPLEMENTATIONS ==========

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average"""
    alpha = 2 / (period + 1)
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = data[i] * alpha + ema[i-1] * (1 - alpha)
    return ema


def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average"""
    sma = np.zeros_like(data)
    cumsum = np.cumsum(data)
    sma[period-1:] = (cumsum[period-1:] - np.concatenate(([0], cumsum[:-period]))) / period
    sma[:period-1] = data[:period-1]
    return sma


def _macd(data: np.ndarray, fast: int, slow: int, signal: int):
    """MACD Indicator"""
    ema_fast = _ema(data, fast)
    ema_slow = _ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _rsi(data: np.ndarray, period: int) -> np.ndarray:
    """Relative Strength Index"""
    delta = np.diff(data, prepend=data[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = _sma(gain, period)
    avg_loss = _sma(loss, period)
    
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _stochastic_close(price: np.ndarray, k_period: int, d_period: int):
    """Stochastic Oscillator using close price only"""
    stoch_k = np.zeros_like(price)
    
    for i in range(k_period - 1, len(price)):
        window = price[i - k_period + 1:i + 1]
        lowest = np.min(window)
        highest = np.max(window)
        if highest != lowest:
            stoch_k[i] = 100 * (price[i] - lowest) / (highest - lowest)
        else:
            stoch_k[i] = 50
    
    stoch_d = _sma(stoch_k, d_period)
    return stoch_k, stoch_d


def _cci_close(price: np.ndarray, period: int) -> np.ndarray:
    """CCI using close price only"""
    tp = price
    sma_tp = _sma(tp, period)
    mean_dev = np.zeros_like(price)
    
    for i in range(period - 1, len(price)):
        window = tp[i - period + 1:i + 1]
        mean_dev[i] = np.mean(np.abs(window - sma_tp[i]))
    
    cci = (tp - sma_tp) / (0.015 * mean_dev + 1e-9)
    return cci


def _roc(data: np.ndarray, period: int) -> np.ndarray:
    """Rate of Change (percentage)"""
    roc = np.zeros_like(data)
    roc[period:] = (data[period:] - data[:-period]) / (data[:-period] + 1e-9) * 100
    roc[:period] = 0
    return roc


def _williams_r_close(price: np.ndarray, period: int) -> np.ndarray:
    """Williams %R using close price only (range -100 to 0)"""
    williams_r = np.zeros_like(price)
    
    for i in range(period - 1, len(price)):
        window = price[i - period + 1:i + 1]
        highest = np.max(window)
        lowest = np.min(window)
        if highest != lowest:
            williams_r[i] = -100 * (highest - price[i]) / (highest - lowest)
        else:
            williams_r[i] = -50
    
    return williams_r


def _mfi(price: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Money Flow Index"""
    typical_price = price
    money_flow = typical_price * volume
    
    positive_flow = np.zeros_like(price)
    negative_flow = np.zeros_like(price)
    
    for i in range(1, len(price)):
        if typical_price[i] > typical_price[i-1]:
            positive_flow[i] = money_flow[i]
        elif typical_price[i] < typical_price[i-1]:
            negative_flow[i] = money_flow[i]
    
    positive_sum = _sma(positive_flow, period)
    negative_sum = _sma(negative_flow, period)
    
    money_ratio = positive_sum / (negative_sum + 1e-9)
    mfi = 100 - (100 / (1 + money_ratio))
    return mfi


def _bollinger_bands(price: np.ndarray, period: int, std_dev: float):
    """Bollinger Bands"""
    middle = _sma(price, period)
    std = np.zeros_like(price)
    
    for i in range(period - 1, len(price)):
        window = price[i - period + 1:i + 1]
        std[i] = np.std(window)
    
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def _atr_close(price: np.ndarray, period: int) -> np.ndarray:
    """ATR using close price only (price change as proxy)"""
    price_change = np.abs(np.diff(price, prepend=price[0]))
    atr = _sma(price_change, period)
    return atr


def _keltner_channel_close(price: np.ndarray, period: int, multiplier: float):
    """Keltner Channel using close price only"""
    middle = _ema(price, period)
    atr = _atr_close(price, period)
    upper = middle + multiplier * atr
    lower = middle - multiplier * atr
    return upper, middle, lower


def _obv(price: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On-Balance Volume"""
    obv = np.zeros_like(price)
    obv[0] = volume[0]
    
    for i in range(1, len(price)):
        if price[i] > price[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif price[i] < price[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    
    return obv


def _vwap(price: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Volume Weighted Average Price (cumulative)"""
    cumulative_pv = np.cumsum(price * volume)
    cumulative_volume = np.cumsum(volume)
    vwap = cumulative_pv / (cumulative_volume + 1e-9)
    return vwap


def _chaikin_money_flow_close(price: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Chaikin Money Flow using close price only"""
    mf_multiplier = np.zeros_like(price)
    for i in range(1, len(price)):
        price_range = abs(price[i] - price[i-1]) + 1e-9
        mf_multiplier[i] = (price[i] - price[i-1]) / price_range
    
    mf_volume = mf_multiplier * volume
    cmf = _sma(mf_volume, period) / (_sma(volume, period) + 1e-9)
    return cmf


def _adx_close(price: np.ndarray, period: int) -> np.ndarray:
    """Simplified ADX using close price only"""
    price_change = np.diff(price, prepend=price[0])
    
    plus_dm = np.where(price_change > 0, price_change, 0)
    minus_dm = np.where(price_change < 0, -price_change, 0)
    
    atr = _atr_close(price, period)
    
    plus_di = 100 * _sma(plus_dm, period) / (atr + 1e-9)
    minus_di = 100 * _sma(minus_dm, period) / (atr + 1e-9)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    adx = _sma(dx, period)
    
    return adx


def _adx_full(price: np.ndarray, period: int):
    """Full ADX with DI+ and DI-"""
    price_change = np.diff(price, prepend=price[0])
    
    plus_dm = np.where(price_change > 0, price_change, 0)
    minus_dm = np.where(price_change < 0, -price_change, 0)
    
    atr = _atr_close(price, period)
    
    plus_di = 100 * _sma(plus_dm, period) / (atr + 1e-9)
    minus_di = 100 * _sma(minus_dm, period) / (atr + 1e-9)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    adx = _sma(dx, period)
    
    return adx, plus_di, minus_di


# ========== EXAMPLE USAGE ==========
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n = 500
    
    # Create synthetic price data (random walk with trend)
    price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    volume = np.random.randint(1000, 10000, n)
    
    # Calculate normalized indicators
    indicators = calculate_best_indicators(price, volume)
    
    print(indicators, len(indicators))
    
    # Print results
    print("=" * 60)
    print("NORMALIZED TRADING INDICATORS")
    print("All indicators normalized to range [-1, 1]")
    print("=" * 60)
    
    for name, value in sorted(indicators.items()):
        # Create visual bar
        bar_len = int(abs(value) * 20)
        if value > 0:
            bar = "█" * bar_len + "░" * (20 - bar_len) + " +"
        elif value < 0:
            bar = "░" * (20 - bar_len) + "█" * bar_len + " -"
        else:
            bar = "░" * 20 + " 0"
        
        print(f"{name:20s}: {value:6.3f} |{bar}|")
    
    print("\n" + "=" * 60)
    print("INTERPRETATION (Normalized to [-1, 1])")
    print("=" * 60)
    
    rsi = indicators.get('rsi', 0)
    if rsi > 0.3:
        print(f"RSI: Bullish momentum ({rsi:+.2f})")
    elif rsi < -0.3:
        print(f"RSI: Bearish momentum ({rsi:+.2f})")
    else:
        print(f"RSI: Neutral ({rsi:+.2f})")
    
    adx = indicators.get('adx', -1)
    if adx > 0.2:
        print(f"ADX: Strong trend ({adx:+.2f})")
    elif adx > -0.2:
        print(f"ADX: Moderate trend ({adx:+.2f})")
    else:
        print(f"ADX: Weak/Ranging ({adx:+.2f})")
    
    macd = indicators.get('macd_histogram', 0)
    if macd > 0.1:
        print(f"MACD: Strongly bullish ({macd:+.2f})")
    elif macd > 0:
        print(f"MACD: Bullish ({macd:+.2f})")
    elif macd > -0.1:
        print(f"MACD: Bearish ({macd:+.2f})")
    else:
        print(f"MACD: Strongly bearish ({macd:+.2f})")
    
    bb_pos = indicators.get('bb_position', 0)
    if bb_pos > 0.8:
        print(f"Bollinger Bands: Near upper band ({bb_pos:+.2f}) - Potential overbought")
    elif bb_pos < -0.8:
        print(f"Bollinger Bands: Near lower band ({bb_pos:+.2f}) - Potential oversold")
    else:
        print(f"Bollinger Bands: Mid-range ({bb_pos:+.2f})")
    
    mfi = indicators.get('mfi', 0)
    if mfi > 0.3:
        print(f"MFI: Money inflow ({mfi:+.2f})")
    elif mfi < -0.3:
        print(f"MFI: Money outflow ({mfi:+.2f})")
    else:
        print(f"MFI: Neutral ({mfi:+.2f})")
    
    volume_ratio = indicators.get('volume_ratio', 0)
    if volume_ratio > 0.3:
        print(f"Volume: Above average ({volume_ratio:+.2f})")
    elif volume_ratio < -0.3:
        print(f"Volume: Below average ({volume_ratio:+.2f})")
    else:
        print(f"Volume: Average ({volume_ratio:+.2f})")