import numpy as np
import pandas as pd

def calculate_atr_from_close(price, period=10):
    """
    Calculate ATR using only price price (simplified)
    
    Uses price change as a proxy for true range
    """
    price = np.array(price)
    price_change = np.abs(np.diff(price, prepend=price[0]))
    
    # Simplified ATR using EMA of price changes
    atr = np.zeros_like(price)
    atr[0] = price_change[0]
    
    for i in range(1, len(price)):
        atr[i] = (atr[i-1] * (period - 1) + price_change[i]) / period
    
    return atr


def calculate_supertrend_close_only(price, period=10, multiplier=3.0):
    """
    Calculate Supertrend using ONLY price price
    
    Parameters:
    -----------
    price : np array
        Close price data
    period : int
        ATR period (default: 10)
    multiplier : float
        ATR multiplier (default: 3.0)
    
    Returns:
    --------
    supertrend : numpy array
        Supertrend values
    trend_direction : numpy array
        1 = uptrend, -1 = downtrend
    """
    
    # Calculate simplified ATR from price price
    atr = calculate_atr_from_close(price, period)
    
    # Basic bands (using price as proxy for hl2)
    basic_upper = price + (multiplier * atr)
    basic_lower = price - (multiplier * atr)
    
    # Initialize
    final_upper = np.zeros_like(price)
    final_lower = np.zeros_like(price)
    supertrend = np.zeros_like(price)
    trend = np.ones_like(price)  # 1 = uptrend, -1 = downtrend
    
    # First element
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    supertrend[0] = final_lower[0]
    
    for i in range(1, len(price)):
        # Final Upper Band
        if basic_upper[i] < final_upper[i-1] or price[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]
        
        # Final Lower Band
        if basic_lower[i] > final_lower[i-1] or price[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]
        
        # Determine Supertrend
        if trend[i-1] == 1:  # Previously uptrend
            if price[i] <= final_lower[i]:
                trend[i] = -1
                supertrend[i] = final_upper[i]
            else:
                trend[i] = 1
                supertrend[i] = final_lower[i]
        else:  # Previously downtrend
            if price[i] >= final_upper[i]:
                trend[i] = 1
                supertrend[i] = final_lower[i]
            else:
                trend[i] = -1
                supertrend[i] = final_upper[i]
    
    return supertrend, trend


def get_signals_close_only(price, period=10, multiplier=3.0):
    """
    Generate trading signals from price-only Supertrend
    """
    _, trend = calculate_supertrend_close_only(price, period, multiplier)
    
    signals = np.zeros_like(trend)
    for i in range(1, len(trend)):
        if trend[i] == 1 and trend[i-1] == -1:
            signals[i] = 1    # BUY
        elif trend[i] == -1 and trend[i-1] == 1:
            signals[i] = -1   # SELL
    
    return signals


# ========== PANDAS VERSION ==========

def supertrend_close_pandas(df, period=10, multiplier=3.0, close_col='price'):
    """
    Calculate Supertrend using only price price (pandas version)
    """
    df = df.copy()
    price = df[close_col].values
    
    supertrend, trend = calculate_supertrend_close_only(price, period, multiplier)
    
    df['supertrend'] = supertrend
    df['trend'] = trend
    df['signal'] = 0
    
    # Generate signals
    df.loc[(df['trend'] == 1) & (df['trend'].shift(1) == -1), 'signal'] = 1
    df.loc[(df['trend'] == -1) & (df['trend'].shift(1) == 1), 'signal'] = -1
    
    return df


# ========== NUMPY VECTORIZED (FASTER) ==========

def supertrend_close_vectorized(price, period=10, multiplier=3.0):
    """
    Vectorized Supertrend using only price price (faster for large arrays)
    """
    price = np.asarray(price)
    n = len(price)
    
    # Price change for ATR
    price_change = np.zeros_like(price)
    price_change[1:] = np.abs(np.diff(price))
    price_change[0] = price_change[1] if n > 1 else 0
    
    # EMA ATR
    alpha = 1 / period
    atr = np.zeros_like(price)
    atr[0] = price_change[0]
    for i in range(1, n):
        atr[i] = (1 - alpha) * atr[i-1] + alpha * price_change[i]
    
    # Bands
    basic_upper = price + multiplier * atr
    basic_lower = price - multiplier * atr
    
    # Initialize
    final_upper = np.zeros_like(price)
    final_lower = np.zeros_like(price)
    supertrend = np.zeros_like(price)
    trend = np.ones_like(price, dtype=int)
    
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    supertrend[0] = final_lower[0]
    
    for i in range(1, n):
        # Upper band
        final_upper[i] = basic_upper[i] if (basic_upper[i] < final_upper[i-1] or price[i-1] > final_upper[i-1]) else final_upper[i-1]
        # Lower band
        final_lower[i] = basic_lower[i] if (basic_lower[i] > final_lower[i-1] or price[i-1] < final_lower[i-1]) else final_lower[i-1]
        
        if trend[i-1] == 1:
            if price[i] <= final_lower[i]:
                trend[i] = -1
                supertrend[i] = final_upper[i]
            else:
                trend[i] = 1
                supertrend[i] = final_lower[i]
        else:
            if price[i] >= final_upper[i]:
                trend[i] = 1
                supertrend[i] = final_lower[i]
            else:
                trend[i] = -1
                supertrend[i] = final_upper[i]
    
    return supertrend, trend


# ========== EXAMPLE ==========

if __name__ == "__main__":
    # Sample price prices
    np.random.seed(42)
    close_prices = 50000 + np.cumsum(np.random.randn(200) * 100)
    
    # Calculate Supertrend
    supertrend, trend = calculate_supertrend_close_only(close_prices, period=10, multiplier=3.0)
    signals = get_signals_close_only(close_prices, period=10, multiplier=3.0)
    
    print("Close Price Supertrend Signals")
    print("-" * 50)
    
    for i in range(-20, 0):
        signal_str = ""
        if signals[i] == 1:
            signal_str = "🔴 BUY"
        elif signals[i] == -1:
            signal_str = "🟢 SELL"
        else:
            signal_str = "⚪ HOLD"
        
        trend_str = "📈 UP" if trend[i] == 1 else "📉 DOWN"
        
        print(f"{i:3d}: Price={close_prices[i]:8.2f} | ST={supertrend[i]:8.2f} | {trend_str} | {signal_str}")
    
    # Latest signal
    if signals[-1] == 1:
        print("\n✅ LATEST SIGNAL: BUY")
    elif signals[-1] == -1:
        print("\n✅ LATEST SIGNAL: SELL")
    else:
        print("\n⏸️ NO SIGNAL - HOLD")