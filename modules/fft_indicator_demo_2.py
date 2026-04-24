import torch
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from Legacy.Binance_new import Binance # Get_Price_Data_By_Time

# Set random seed for reproducibility
torch.manual_seed(42)

#--------------------------------------- Get market history -----------------------------------
bn = Binance.instantiate(connect=False)
# Get_Price_Data_By_Time function seamlessly collects/creates price candles from Binance.
datatype = "klines" # "aggTrades" #, "klines"
symbol = "BTCUSDT"      # Any pair listed on Binance.
start = datetime(2025, 1, 1)    # Any datetime. Note Binance started operation in July 2017.
end = datetime(2026, 1, 1) # + timedelta(minutes=100)     # Any datetime
interval = '1m'    # in '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d'.  (excludes: '3d', '1w')
candles, nFills, successful, _, reason = bn.Get_Price_Data_By_Time(datatype, symbol, interval, start, end)
# "Returning 10 columns: O H L C V, Quote V, nTrades, Buy base V, Buy quote V, Mark (0: original not forged)"
open_idx, high_idx, low_idx, close_idx, volume_idx, mark_idx = 0, 1, 2, 3, 4, 9
if successful:          # price is a numpy array.
    print("\n Successful !!!!!!!!!! (nCandles, nAttributes) = {}, nForged = {}".format(candles.shape, nFills))
else:
    print("\n Failed to Get_Price_Data_By_Time", reason)
    exit(0)
    

def generate_sample_price_data(n_bars=500):

    price = candles[:1000, close_idx]
    norm_price = (price - price.mean()) / price.std()
    signal = torch.from_numpy(norm_price).float()
    
    return signal[:n_bars]

class FFTIndicators:
    def __init__(self, price, lookback=200):
        # Convert to torch tensor if numpy array
        if isinstance(price, np.ndarray):
            price = torch.from_numpy(price).float()
        self.price = price
        self.lookback = lookback
        self.detrended = None
        self.freqs = None
        self.magnitude = None
        self.phase = None
        self._compute_fft()
    
    def _compute_fft(self):
        """Compute FFT with proper detrending"""
        # Take last lookback bars
        signal = self.price[-self.lookback:]
        
        # Detrend using linear regression (pure PyTorch)
        t = torch.arange(len(signal), dtype=torch.float32)
        t_mean = t.mean()
        signal_mean = signal.mean()
        
        a = ((t - t_mean) * (signal - signal_mean)).sum() / ((t - t_mean)**2).sum()
        b = signal_mean - a * t_mean
        trend = a * t + b
        self.detrended = signal - trend
        
        # Apply Hann window
        windowed = self.detrended * torch.hann_window(len(signal))
        
        # Compute FFT
        n = len(signal)
        fft_result = torch.fft.fft(windowed)
        
        # Get frequencies (cycles per bar)
        self.freqs = torch.fft.fftfreq(n, d=1.0)[:n//2]
        self.magnitude = torch.abs(fft_result[:n//2])
        self.phase = torch.angle(fft_result[:n//2])
        
        # Remove DC component
        self.magnitude[0] = 0
        
        # Normalize magnitude
        if self.magnitude.sum() > 0:
            self.magnitude = self.magnitude / self.magnitude.sum()
    
    def indicator_1_magnitude_spectrum(self):
        """Indicator 1: FFT Magnitude Spectrum / Periodogram"""
        # Filter out zero frequencies
        mask = self.freqs > 0
        periods = 1.0 / self.freqs[mask]
        magnitudes = self.magnitude[mask]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Periodogram
        axes[0].bar(periods.numpy(), magnitudes.numpy(), width=2, alpha=0.7)
        axes[0].set_xlabel('Period (bars)')
        axes[0].set_ylabel('Strength')
        axes[0].set_title('Cycle Strength Periodogram')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(magnitudes.mean().item(), color='r', linestyle='--', alpha=0.5, label='Mean strength')
        axes[0].legend()
        
        # Top cycles
        top_indices = magnitudes.argsort(descending=True)[:5]
        top_periods = periods[top_indices]
        top_strengths = magnitudes[top_indices]
        
        axes[1].barh(range(len(top_periods)), top_strengths.numpy(), alpha=0.7)
        axes[1].set_yticks(range(len(top_periods)))
        axes[1].set_yticklabels([f"{p:.1f} bars" for p in top_periods.numpy()])
        axes[1].set_xlabel('Strength')
        axes[1].set_title('Top 5 Dominant Cycles')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return dict(zip(top_periods.numpy(), top_strengths.numpy()))
    
    def indicator_2_cycle_strength_percentile(self, historical_windows=50):
        """Indicator 2: Cycle Strength Percentile Bands"""
        
        # Simulate historical magnitudes
        np.random.seed(42)
        historical_magnitudes = []
        
        for i in range(historical_windows):
            # Generate synthetic historical magnitudes
            hist_mag = self.magnitude * (0.5 + 0.5 * torch.randn(len(self.magnitude)))
            hist_mag = hist_mag / hist_mag.sum()
            historical_magnitudes.append(hist_mag)
        
        historical_magnitudes = torch.stack(historical_magnitudes)
        
        # Calculate percentiles
        p25 = historical_magnitudes.quantile(0.25, dim=0)
        p75 = historical_magnitudes.quantile(0.75, dim=0)
        
        mask = self.freqs > 0
        periods = 1.0 / self.freqs[mask]
        current_mag = self.magnitude[mask]
        p25_vals = p25[mask]
        p75_vals = p75[mask]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Percentile bands
        axes[0].fill_between(periods.numpy(), p25_vals.numpy(), p75_vals.numpy(), 
                              alpha=0.3, label='25th-75th Percentile')
        axes[0].plot(periods.numpy(), current_mag.numpy(), 'r-', linewidth=2, label='Current')
        axes[0].set_xlabel('Period (bars)')
        axes[0].set_ylabel('Cycle Strength')
        axes[0].set_title('Cycle Strength Percentile Bands')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Regime indicator
        above_75 = (current_mag > p75_vals).float()
        below_25 = (current_mag < p25_vals).float()
        
        regime_score = (above_75.sum() - below_25.sum()).item() / len(current_mag)
        
        axes[1].bar(['Cycle Strength', 'Regime'], 
                    [current_mag.mean().item(), regime_score],
                    color=['blue', 'green' if regime_score > 0 else 'red'])
        axes[1].set_ylabel('Score')
        axes[1].set_title(f'Market Regime: {"Cyclical" if regime_score > 0 else "Trendless/Choppy"}')
        axes[1].axhline(0, color='black', linestyle='-', linewidth=0.5)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'regime_score': regime_score,
            'is_cyclical': regime_score > 0,
            'above_75_percent': above_75.sum().item() / len(current_mag),
            'below_25_percent': below_25.sum().item() / len(current_mag)
        }
    
    def indicator_3_fourier_extrapolator(self, n_components=8, forecast_bars=50):
        """Indicator 3: Fourier Extrapolator / Forecast"""
        
        n = len(self.detrended)
        t = torch.arange(n)
        
        # Get top frequency components
        top_indices = self.magnitude.argsort(descending=True)[:n_components]
        top_freqs = self.freqs[top_indices]
        top_mags = self.magnitude[top_indices]
        top_phases = self.phase[top_indices]
        
        # Reconstruct signal
        reconstructed = torch.zeros(n)
        for freq, mag, phase in zip(top_freqs, top_mags, top_phases):
            reconstructed += mag * torch.cos(2 * torch.pi * freq * t + phase)
        
        # Forecast
        forecast_t = torch.arange(n, n + forecast_bars)
        forecast = torch.zeros(forecast_bars)
        for freq, mag, phase in zip(top_freqs, top_mags, top_phases):
            forecast += mag * torch.cos(2 * torch.pi * freq * forecast_t + phase)
        
        # Normalize
        reconstructed = reconstructed * (self.detrended.std() / reconstructed.std())
        forecast = forecast * (self.detrended.std() / reconstructed.std())
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Original vs reconstructed
        axes[0].plot(self.detrended.numpy(), label='Original (Detrended)', alpha=0.7)
        axes[0].plot(reconstructed.numpy(), label=f'Reconstructed ({n_components} components)', alpha=0.7)
        axes[0].set_title('Signal Reconstruction from Fourier Components')
        axes[0].set_xlabel('Bar')
        axes[0].set_ylabel('Amplitude')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Forecast
        full_t = torch.cat([t, forecast_t])
        full_signal = torch.cat([self.detrended, torch.full((forecast_bars,), float('nan'))])
        full_forecast = torch.cat([torch.full((n,), float('nan')), forecast])
        
        axes[1].plot(full_t.numpy(), full_signal.numpy(), label='Historical', linewidth=1)
        axes[1].plot(full_t.numpy(), full_forecast.numpy(), 'r--', label='Forecast', linewidth=2)
        axes[1].axvline(x=n, color='g', linestyle='--', alpha=0.5, label='Forecast Start')
        axes[1].set_title('Fourier Extrapolation Forecast')
        axes[1].set_xlabel('Bar')
        axes[1].set_ylabel('Amplitude')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Add confidence bands
        error_std = (self.detrended - reconstructed).std()
        upper_band = forecast + 2 * error_std
        lower_band = forecast - 2 * error_std
        
        full_upper = torch.cat([torch.full((n,), float('nan')), upper_band])
        full_lower = torch.cat([torch.full((n,), float('nan')), lower_band])
        
        axes[1].fill_between(full_t[n:].numpy(), lower_band.numpy(), upper_band.numpy(), 
                              alpha=0.2, color='red', label='95% Confidence')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
        
        return {
            'forecast': forecast.numpy(),
            'confidence_upper': upper_band.numpy(),
            'confidence_lower': lower_band.numpy(),
            'reconstruction_error': error_std.item()
        }
    
    def indicator_4_detrended_oscillator(self, smoothing=3):
        """Indicator 4: FFT Detrended Oscillator"""
        
        # Reconstruct using top cycles for oscillator
        n_components = 5
        top_indices = self.magnitude.argsort(descending=True)[:n_components]
        top_freqs = self.freqs[top_indices]
        top_mags = self.magnitude[top_indices]
        top_phases = self.phase[top_indices]
        
        t = torch.arange(len(self.detrended))
        oscillator = torch.zeros(len(self.detrended))
        for freq, mag, phase in zip(top_freqs, top_mags, top_phases):
            oscillator += mag * torch.sin(2 * torch.pi * freq * t + phase)
        
        # Normalize to -1 to 1 range
        oscillator = oscillator / (oscillator.abs().max() + 1e-6)
        
        # Apply smoothing
        kernel = torch.ones(smoothing) / smoothing
        oscillator_smoothed = torch.conv1d(oscillator.view(1, 1, -1), 
                                           kernel.view(1, 1, -1), 
                                           padding='same').squeeze()
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Oscillator
        axes[0].plot(oscillator.numpy(), label='Raw Oscillator', alpha=0.5)
        axes[0].plot(oscillator_smoothed.numpy(), label='Smoothed', linewidth=2)
        axes[0].axhline(0, color='black', linestyle='-', linewidth=0.5)
        axes[0].axhline(0.5, color='g', linestyle='--', alpha=0.5)
        axes[0].axhline(-0.5, color='r', linestyle='--', alpha=0.5)
        axes[0].set_title('FFT Detrended Oscillator')
        axes[0].set_ylabel('Oscillator Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Color background based on direction
        for i in range(1, len(oscillator_smoothed)):
            color = 'green' if oscillator_smoothed[i] > oscillator_smoothed[i-1] else 'red'
            axes[1].axvspan(i-1, i, alpha=0.3, color=color)
        
        axes[1].plot(oscillator_smoothed.numpy(), 'b-', linewidth=2)
        axes[1].axhline(0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_title('Oscillator Direction (Green=Up, Red=Down)')
        axes[1].set_xlabel('Bar')
        axes[1].set_ylabel('Oscillator Value')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Generate trading signals
        signals = []
        for i in range(1, len(oscillator_smoothed)):
            if oscillator_smoothed[i] > 0 and oscillator_smoothed[i-1] <= 0:
                signals.append(('BUY', i))
            elif oscillator_smoothed[i] < 0 and oscillator_smoothed[i-1] >= 0:
                signals.append(('SELL', i))
        
        return {
            'oscillator': oscillator_smoothed.numpy(),
            'signals': signals,
            'current_value': oscillator_smoothed[-1].item(),
            'current_direction': 'UP' if oscillator_smoothed[-1] > oscillator_smoothed[-2] else 'DOWN'
        }
    
    def indicator_5_interpreted_fft(self):
        """Indicator 5: Interpreted FFT (Top 3 Cycles)"""
        
        # Get top 3 cycles
        mask = self.freqs > 0
        periods = 1.0 / self.freqs[mask]
        magnitudes = self.magnitude[mask]
        
        top_indices = magnitudes.argsort(descending=True)[:3]
        top_periods = periods[top_indices]
        top_strengths = magnitudes[top_indices]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Display top cycles
        y_pos = np.arange(3)
        axes[0].barh(y_pos, top_strengths.numpy(), alpha=0.7)
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels([f"{p:.1f} bars" for p in top_periods.numpy()])
        axes[0].set_xlabel('Strength')
        axes[0].set_title('Top 3 Dominant Cycles')
        axes[0].grid(True, alpha=0.3)
        
        # Add strength values
        for i, (period, strength) in enumerate(zip(top_periods, top_strengths)):
            axes[0].text(strength.item() + 0.01, i, f'{strength:.3f}', va='center')
        
        # Recommendation based on cycles
        primary_cycle = top_periods[0].item()
        
        if primary_cycle < 30:
            timeframe = "Short-term / Intraday"
            strategy = "Scalping / Day Trading"
        elif primary_cycle < 100:
            timeframe = "Medium-term / Swing"
            strategy = "Swing Trading"
        else:
            timeframe = "Long-term / Position"
            strategy = "Position Trading / Trend Following"
        
        # Parameter recommendations
        recommendations = [
            f"Primary Cycle: {primary_cycle:.1f} bars",
            f"Recommended Timeframe: {timeframe}",
            f"Suggested Strategy: {strategy}",
            f"MA Period Suggestion: {int(primary_cycle)}",
            f"RSI Period Suggestion: {max(14, int(primary_cycle/3))}",
        ]
        
        axes[1].axis('tight')
        axes[1].axis('off')
        axes[1].set_title('Trading Recommendations')
        y_pos_text = 0.9
        for rec in recommendations:
            axes[1].text(0.1, y_pos_text, rec, transform=axes[1].transAxes,
                        fontsize=12, verticalalignment='top')
            y_pos_text -= 0.15
        
        plt.tight_layout()
        plt.show()
        
        return {
            'primary_cycle': primary_cycle,
            'secondary_cycle': top_periods[1].item() if len(top_periods) > 1 else None,
            'tertiary_cycle': top_periods[2].item() if len(top_periods) > 2 else None,
            'recommended_ma': int(primary_cycle),
            'recommended_rsi': max(14, int(primary_cycle/3))
        }


# ============ COMPLETE DEMONSTRATION ============

def complete_fft_demo():
    """Run complete demonstration of all FFT indicators"""
    
    print("=" * 60)
    print("FFT-BASED TRADING INDICATORS DEMONSTRATION")
    print("=" * 60)
    
    # Generate sample price data
    print("\n1. Generating sample price data with 60, 30, and 15-day cycles...")
    price = generate_sample_price_data(2000)
    
    # Plot raw price
    plt.figure(figsize=(14, 4))
    plt.plot(price)
    plt.title('Sample Price Data with Multiple Cycles')
    plt.xlabel('Bar')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Initialize indicators
    print("\n2. Initializing FFT indicators...")
    indicators = FFTIndicators(price, lookback=200)
    
    # Indicator 1: Magnitude Spectrum
    print("\n" + "=" * 40)
    print("INDICATOR 1: FFT Magnitude Spectrum")
    print("=" * 40)
    top_cycles = indicators.indicator_1_magnitude_spectrum()
    print(f"Top cycles detected: {top_cycles}")
    
    # Indicator 2: Cycle Strength Percentile Bands
    print("\n" + "=" * 40)
    print("INDICATOR 2: Cycle Strength Percentile Bands")
    print("=" * 40)
    regime = indicators.indicator_2_cycle_strength_percentile()
    print(f"Market Regime: {'Cyclical' if regime['is_cyclical'] else 'Trendless'}")
    print(f"Regime Score: {regime['regime_score']:.3f}")
    
    # Indicator 3: Fourier Extrapolator
    print("\n" + "=" * 40)
    print("INDICATOR 3: Fourier Extrapolator (Forecast)")
    print("=" * 40)
    forecast = indicators.indicator_3_fourier_extrapolator(n_components=8, forecast_bars=50)
    print(f"Forecast for next 50 bars: Mean={forecast['forecast'].mean():.3f}")
    print(f"Reconstruction Error: {forecast['reconstruction_error']:.3f}")
    
    # Indicator 4: Detrended Oscillator
    print("\n" + "=" * 40)
    print("INDICATOR 4: FFT Detrended Oscillator")
    print("=" * 40)
    oscillator = indicators.indicator_4_detrended_oscillator()
    print(f"Current Oscillator Value: {oscillator['current_value']:.3f}")
    print(f"Current Direction: {oscillator['current_direction']}")
    print(f"Recent Signals: {oscillator['signals'][-5:]}")
    
    # Indicator 5: Interpreted FFT
    print("\n" + "=" * 40)
    print("INDICATOR 5: Interpreted FFT (Top 3 Cycles)")
    print("=" * 40)
    recommendations = indicators.indicator_5_interpreted_fft()
    print(f"Primary Cycle: {recommendations['primary_cycle']:.1f} bars")
    print(f"Recommended MA Period: {recommendations['recommended_ma']}")
    print(f"Recommended RSI Period: {recommendations['recommended_rsi']}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    
    return indicators

# Run the complete demo
if __name__ == "__main__":
    indicators = complete_fft_demo()